"""
OMR Quiz 채점 웹 앱 - 원스톱 OMR 인식 및 CSV 결과 다운로드
Flask 기반, QUIZ-3 template 고정 사용
답안 키 CSV 업로드 시 자동 채점 지원 + 디버그 이미지 확인
"""

import os
import uuid
import base64
import collections
from pathlib import Path
from io import StringIO, BytesIO

import cv2
import numpy as np
import pandas as pd
from flask import (
    Flask,
    render_template_string,
    request,
    send_file,
    jsonify,
)

from src.template import Template
from src.defaults import CONFIG_DEFAULTS
from src.utils.parsing import get_concatenated_response
from src.utils.image import ImageUtils
from src.logger import logger

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max

# ── 고정 템플릿 경로 (QUIZ-3) ──
TEMPLATE_PATH = Path(__file__).parent / "template" / "template.json"

_template = None


def get_template():
    global _template
    if _template is None:
        if not TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"template.json not found at {TEMPLATE_PATH}. "
                "template/template.json 파일이 필요합니다."
            )

        # Levels 대비 강화: low를 올려서 인쇄 배경(연한 회색)을 흰색으로 제거
        # 기본 low=0.3 -> 0.45 로 변경하여 마킹만 남김
        from copy import deepcopy
        tuning = deepcopy(CONFIG_DEFAULTS)
        _template = Template(TEMPLATE_PATH, tuning)

        # preProcessor 중 Levels의 gamma LUT을 재생성
        for pp in _template.pre_processors:
            if pp.__class__.__name__ == "Levels":
                low_val = int(255 * 0.55)   # 공격적 밝기 컷: 연한 잔상 완전 제거
                high_val = int(255 * 0.70)   # 진한 마킹만 어둡게 유지
                gamma = 1.0
                inv_gamma = 1.0 / gamma
                new_lut = np.array([
                    0 if i <= low_val else (
                        255 if i >= high_val else
                        int((((i - low_val) / (high_val - low_val)) ** inv_gamma) * 255)
                    )
                    for i in range(256)
                ], dtype="uint8")
                pp.gamma = new_lut
                logger.info(f"Levels 대비 강화: low={low_val/255:.2f}, high={high_val/255:.2f}")
                break

    return _template


# ─────────── 이미지 처리 유틸 ───────────


def auto_rotate_image(image, template=None):
    """세로 이미지 -> 가로로 자동 회전 (양방향 시도 후 최적 선택)"""
    h, w = image.shape[:2]
    if w >= h:
        return image, False

    if template is not None:
        rotated_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        score_cw = _evaluate_rotation_quality(template, rotated_cw)
        score_ccw = _evaluate_rotation_quality(template, rotated_ccw)
        if score_cw >= score_ccw:
            return rotated_cw, True
        else:
            return rotated_ccw, True
    else:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), True


def _evaluate_rotation_quality(template, image):
    """
    회전 품질을 평가하는 함수.
    마커 검출 성공 시 보너스를 주고, 학번(Num) 필드의 인식률을 추가로 고려.
    """
    try:
        # 마커 검출 시도
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 간단한 마커 검출 시도 (성공 여부만 확인)
        marker_bonus = 0
        try:
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 19, 2
            )
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            squares = []
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    # 마커 크기 범위 확대 (다양한 스캐너 해상도 지원)
                    if 30 <= w <= 120 and 30 <= h <= 120:
                        aspect_ratio = max(w, h) / min(w, h)
                        # 정사각형에 가까운 것만 선택
                        if aspect_ratio <= 1.5:
                            squares.append(approx)
            # 마커가 2개 이상 검출되면 보너스
            if len(squares) >= 2:
                marker_bonus = 20
        except Exception:
            pass
        
        # OMR 응답 인식 시도
        template.image_instance_ops.reset_all_save_img()
        template.image_instance_ops.append_save_img(1, image)
        processed = template.image_instance_ops.apply_preprocessors(
            "rotation_test", image, template
        )
        if processed is None:
            return -1 + marker_bonus
        
        response_dict, _, multi_marked, _ = (
            template.image_instance_ops.read_omr_response(
                template, image=processed, name="rotation_test", save_dir=None
            )
        )
        omr_response = get_concatenated_response(response_dict, template)
        
        # 학번 필드 인식 시 추가 보너스
        num_field_bonus = 0
        if 'Num' in omr_response and omr_response['Num'] and len(str(omr_response['Num'])) >= 5:
            num_field_bonus = 15
        
        # 전체 응답 중 비어있지 않은 필드 수
        non_empty = sum(1 for v in omr_response.values() if v and v.strip())
        
        # 다중 마킹 패널티
        penalty = 10 if multi_marked else 0
        
        return non_empty + marker_bonus + num_field_bonus - penalty
    except Exception as e:
        logger.debug(f"회전 품질 평가 중 에러: {e}")
        return -1


def increase_contrast_with_clahe(image):
    """CLAHE를 사용한 대비 증가 (rotate_test.py 원본 로직)"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def rotate_image_warp(image, angle):
    """회전 (rotate_test.py 원본 로직)"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h))


def detect_markers(gray_image):
    """
    이미지에서 3개의 사각형 마커를 검출하여 중심점을 반환.
    Returns: (centers_array, success_bool)
    """
    try:
        # CLAHE 대비 증가
        detect_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR) if len(gray_image.shape) == 2 else gray_image.copy()
        detect_img = increase_contrast_with_clahe(detect_img)
        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        
        # 적응형 이진화로 사각형 검출
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 2
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        squares = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                # 마커 크기 범위 확대 (다양한 스캐너 해상도 지원)
                if 30 <= w <= 120 and 30 <= h <= 120:
                    aspect_ratio = max(w, h) / min(w, h)
                    # 정사각형에 가까운 것만 선택 (1:1.5 비율까지 허용)
                    if aspect_ratio <= 1.5:
                        squares.append(approx)
        
        if len(squares) != 3:
            return None, False
        
        # 마커 중심점 계산
        centers = []
        for square in squares:
            M = cv2.moments(square)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])
        
        if len(centers) != 3:
            return None, False
        
        return np.array(centers), True
    except Exception as e:
        logger.debug(f"마커 검출 중 에러: {e}")
        return None, False


def determine_orientation_from_markers(centers):
    """
    마커 위치를 기반으로 이미지 회전 각도를 결정.
    마커가 왼쪽 위, 오른쪽 위, 오른쪽 아래에 오도록 회전 각도 반환.
    
    Returns: rotation_angle (0, 90, 180, 270)
    """
    # Y좌표로 정렬하여 위쪽과 아래쪽 구분
    sorted_by_y = sorted(centers, key=lambda c: c[1])
    
    # Y 좌표 차이 계산
    y_diff_top = abs(sorted_by_y[1][1] - sorted_by_y[0][1])
    y_diff_bottom = abs(sorted_by_y[2][1] - sorted_by_y[1][1])
    
    # 위쪽 2개가 비슷한 높이에 있는지 확인 (임계값: 이미지 높이의 10%)
    image_diagonal = np.sqrt(centers[:, 0].max()**2 + centers[:, 1].max()**2)
    threshold = image_diagonal * 0.1
    
    if y_diff_top < threshold and y_diff_bottom > threshold:
        # 정상: 위쪽 2개, 아래쪽 1개
        top_two = sorted(sorted_by_y[:2], key=lambda c: c[0])
        bottom_one = sorted_by_y[2]
        
        left_x = top_two[0][0]
        right_x = top_two[1][0]
        bottom_x = bottom_one[0]
        
        # 아래쪽 마커가 오른쪽에 있어야 정상 (오른쪽 아래)
        if bottom_x > (left_x + right_x) * 0.6:  # 오른쪽 60% 이상
            return 0
        else:
            return 180  # 좌우 반전
    
    elif y_diff_bottom < threshold and y_diff_top > threshold:
        # 180도 회전: 아래쪽 2개, 위쪽 1개
        return 180
    else:
        # 가로로 3개 배치 → 90도 또는 270도 회전 필요
        sorted_by_x = sorted(centers, key=lambda c: c[0])
        x_diff_left = abs(sorted_by_x[1][0] - sorted_by_x[0][0])
        x_diff_right = abs(sorted_by_x[2][0] - sorted_by_x[1][0])
        
        if x_diff_left < threshold and x_diff_right > threshold:
            # 왼쪽 2개, 오른쪽 1개 → 270도 (시계반대방향)
            return 270
        else:
            # 오른쪽 2개, 왼쪽 1개 → 90도 (시계방향)
            return 90


def detect_markers(gray_image):
    """
    이미지에서 3개의 사각형 마커를 검출하여 중심점을 반환.
    Returns: (centers_array, success_bool)
    """
    try:
        # CLAHE 대비 증가
        detect_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR) if len(gray_image.shape) == 2 else gray_image.copy()
        detect_img = increase_contrast_with_clahe(detect_img)
        gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
        
        # 적응형 이진화로 사각형 검출
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 2
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        squares = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                # 마커 크기 범위 확대 (다양한 스캐너 해상도 지원)
                if 30 <= w <= 120 and 30 <= h <= 120:
                    aspect_ratio = max(w, h) / min(w, h)
                    # 정사각형에 가까운 것만 선택 (1:1.5 비율까지 허용)
                    if aspect_ratio <= 1.5:
                        squares.append(approx)
        
        if len(squares) != 3:
            return None, False
        
        # 마커 중심점 계산
        centers = []
        for square in squares:
            M = cv2.moments(square)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])
        
        if len(centers) != 3:
            return None, False
        
        return np.array(centers), True
    except Exception as e:
        logger.debug(f"마커 검출 중 에러: {e}")
        return None, False


def determine_orientation_from_markers(centers):
    """
    마커 위치를 기반으로 이미지 회전 각도를 결정.
    마커가 왼쪽 위, 오른쪽 위, 오른쪽 아래에 오도록 회전 각도 반환.
    
    Returns: rotation_angle (0, 90, 180, 270)
    """
    # 위쪽 2개, 아래쪽 1개 분류
    sorted_by_y = sorted(centers, key=lambda c: c[1])
    
    # 가장 위쪽 2개
    top_two = sorted(sorted_by_y[:2], key=lambda c: c[0])
    bottom_one = sorted_by_y[2]
    
    # 아래쪽 마커의 x 좌표 확인
    left_x = top_two[0][0]
    right_x = top_two[1][0]
    bottom_x = bottom_one[0]
    
    # 아래쪽 마커가 오른쪽에 있어야 정상 (오른쪽 아래)
    if bottom_x > (left_x + right_x) / 2:
        # 정상 방향
        return 0
    elif bottom_one[1] < top_two[0][1]:
        # 180도 회전 필요 (위아래 뒤집힘)
        return 180
    elif bottom_x < left_x:
        # 90도 시계반대방향 회전 필요
        return 270
    else:
        # 90도 시계방향 회전 필요
        return 90


def marker_align_and_crop(image, color_image=None):
    """
    3개의 코너 마커를 검출하여 Affine 변환으로 정렬+크롭.
    
    올바른 처리 순서:
    1. 컬러 드롭아웃 (빨간 인쇄 제거) → 흑백 변환
    2. 흑백 이미지에서 마커 검출 (4방향 시도)
    3. 마커 위치 기반 회전 보정
    4. 마커 바깥쪽 기준 Affine 변환 + 크롭 → (3507, 2480)
    5. 전처리

    image: 그레이스케일 (Red 채널)
    color_image: 원본 컬러 이미지
    Returns: (warped_grayscale, success_bool, debug_images_dict)
    """
    DST_SIZE = (3507, 2480)
    debug_images = {}

    try:
        # 컬러 이미지가 있으면 사용, 없으면 그레이스케일에서 변환
        if color_image is not None:
            color_img = color_image.copy()
        elif len(image.shape) == 2:
            color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            color_img = image.copy()
        
        # 디버그: 원본 이미지 저장
        debug_images['01_original'] = color_img.copy()
        
        # 1. 컬러 드롭아웃 (빨간 인쇄 제거): max(B, G, R) → 검정 마킹만 남김
        if len(color_img.shape) == 3:
            b, g, r = cv2.split(color_img)
            gray = np.maximum(np.maximum(b, g), r)
        else:
            gray = color_img.copy()
        
        debug_images['02_color_dropout'] = gray.copy()
        logger.info(f"컬러 드롭아웃 완료: {color_img.shape} → {gray.shape}")
        
        # 2. 마커 검출 시도 (최대 4방향)
        best_angle = None
        best_centers = None
        best_rotated_gray = None
        
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                test_img = gray.copy()
            else:
                if angle == 90:
                    test_img = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    test_img = cv2.rotate(gray, cv2.ROTATE_180)
                else:  # 270
                    test_img = cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            centers, success = detect_markers(test_img)
            if success:
                # 마커 위치 기반으로 올바른 방향인지 확인
                marker_angle = determine_orientation_from_markers(centers)
                logger.info(f"각도 {angle}도에서 마커 검출 성공, 필요한 추가 회전: {marker_angle}도")
                
                if marker_angle == 0:
                    # 올바른 방향 찾음
                    best_angle = angle
                    best_centers = centers
                    best_rotated_gray = test_img
                    logger.info(f"✓ 최종 선택: {angle}도 회전으로 정상 방향")
                    break
                elif best_angle is None:
                    # 첫 번째로 마커가 검출된 각도 저장 (fallback)
                    best_angle = angle
                    best_centers = centers
                    best_rotated_gray = test_img
                    logger.info(f"임시 저장: {angle}도 (추가 회전 {marker_angle}도 필요)")
        
        if best_angle is None or best_centers is None:
            logger.warning("모든 방향에서 마커 검출 실패")
            return None, False, debug_images
        
        # 3. 회전 보정 (흑백 이미지)
        rotated_gray = best_rotated_gray
        logger.info(f"회전 보정 적용: {best_angle}도, 크기: {rotated_gray.shape}")
        debug_images['03_rotated'] = rotated_gray.copy()
        
        # 4. 회전된 이미지에서 마커 재검출 (더 정확한 좌표)
        centers, success = detect_markers(rotated_gray)
        if not success:
            logger.warning("회전 후 마커 재검출 실패 - 이전 좌표 사용")
            centers = best_centers
        
        # 마커 위치 표시 (디버그용)
        marked_img = cv2.cvtColor(rotated_gray, cv2.COLOR_GRAY2BGR)
        for i, center in enumerate(centers):
            cv2.circle(marked_img, tuple(center.astype(int)), 20, (0, 255, 0), 3)
            cv2.putText(marked_img, f"M{i+1}", tuple(center.astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        debug_images['04_markers_detected'] = marked_img
        
        # 5. 마커를 왼쪽 위, 오른쪽 위, 오른쪽 아래로 정렬
        sorted_by_y = sorted(centers, key=lambda c: c[1])
        top_two = sorted(sorted_by_y[:2], key=lambda c: c[0])
        bottom_one = sorted_by_y[2]
        
        top_left = top_two[0]
        top_right = top_two[1]
        bottom_right = bottom_one
        
        # 6. Affine 변환으로 워프 (마커 바깥쪽을 기준으로 크롭)
        src_pts = np.array(
            [top_left, top_right, bottom_right], dtype="float32"
        )
        dst_pts = np.array(
            [[0, 0], [DST_SIZE[0], 0], [DST_SIZE[0], DST_SIZE[1]]],
            dtype="float32",
        )

        M_affine = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(rotated_gray, M_affine, DST_SIZE)
        
        logger.info(f"Affine 변환 완료: {rotated_gray.shape} → {warped.shape}, 기대값: {DST_SIZE}")
        debug_images['05_warped'] = warped.copy()

        logger.info(
            f"마커 정렬+크롭 완료 (회전: {best_angle}도): {gray.shape} -> {warped.shape}"
        )
        return warped, True, debug_images

    except Exception as e:
        logger.warning(f"마커 정렬+크롭 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, False, debug_images


def draw_template_bubbles_on_image(image, template):
    """
    이미지에 템플릿의 모든 버블 위치를 표시.
    Returns: 버블이 표시된 이미지 (numpy array)
    """
    try:
        # 이미지 복사
        if len(image.shape) == 2:
            viz_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            viz_img = image.copy()
        
        # 각 field block의 버블 그리기
        for field_block in template.field_blocks:
            for field_bubbles in field_block.traverse_bubbles:
                for bubble in field_bubbles:
                    # 버블 중심점
                    cx, cy = bubble.x, bubble.y
                    
                    # 버블 크기
                    bubble_w, bubble_h = template.bubble_dimensions
                    
                    # 사각형으로 버블 영역 표시 (반투명 녹색)
                    top_left = (int(cx - bubble_w//2), int(cy - bubble_h//2))
                    bottom_right = (int(cx + bubble_w//2), int(cy + bubble_h//2))
                    
                    # 버블 테두리 (녹색)
                    cv2.rectangle(viz_img, top_left, bottom_right, (0, 255, 0), 1)
                    
                    # 중심점 (작은 빨간 점)
                    cv2.circle(viz_img, (cx, cy), 2, (0, 0, 255), -1)
        
        # Field block 이름 표시
        for field_block in template.field_blocks:
            if field_block.traverse_bubbles:
                first_bubble = field_block.traverse_bubbles[0][0]
                cx, cy = first_bubble.x, first_bubble.y
                
                # 텍스트 배경
                text = field_block.name
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                
                # 배경 사각형 (반투명 흰색)
                cv2.rectangle(viz_img, 
                            (cx - 5, cy - text_h - 10), 
                            (cx + text_w + 5, cy - 5),
                            (255, 255, 255), -1)
                
                # 텍스트 (검정색)
                cv2.putText(viz_img, text, (cx, cy - 8),
                           font, font_scale, (0, 0, 0), thickness)
        
        return viz_img
    except Exception as e:
        logger.error(f"템플릿 버블 표시 중 에러: {e}")
        return image


def ensure_standard_size(image, target_size=(3507, 2480)):
    """
    이미지를 표준 크기로 리사이즈 (마커 검출 실패 시 fallback).
    aspect ratio를 유지하면서 리사이즈한 후 패딩 추가.
    """
    try:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 이미 표준 크기면 그대로 반환
        if (w, h) == target_size:
            logger.info(f"이미지가 이미 표준 크기 {target_size}")
            return image, True
        
        # aspect ratio 계산
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)
        
        # 리사이즈
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 패딩 추가 (중앙 정렬)
        if len(image.shape) == 3:
            result = np.full((target_h, target_w, image.shape[2]), 255, dtype=np.uint8)
        else:
            result = np.full((target_h, target_w), 255, dtype=np.uint8)
        
        # 중앙에 배치
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        logger.warning(f"마커 없이 리사이즈: {w}×{h} → {target_w}×{target_h} (scale={scale:.3f})")
        return result, False
        
    except Exception as e:
        logger.error(f"리사이즈 중 에러: {e}")
        return image, False


def process_single_image(template, image, file_name, enable_crop=True, color_image=None):
    """
    단일 이미지 OMR 처리.
    enable_crop=True 시 3개 코너 마커 기반으로 정렬+크롭 수행.
    color_image: 원본 컬러 이미지 (크롭 시 Red 채널 드롭아웃용)
    Returns: (result_dict, error_message)
      result_dict에 final_marked (numpy), debug_images 포함
    """
    try:
        # 마커 기반 정렬+크롭 (개선된 로직)
        was_cropped = False
        debug_images = {}
        
        if enable_crop:
            cropped, success, debug_imgs = marker_align_and_crop(image, color_image=color_image)
            debug_images.update(debug_imgs)
            if success and cropped is not None:
                image = cropped
                was_cropped = True
                logger.info("✓ 마커 기반 정렬 성공")
            else:
                # 마커 검출 실패 시 강제 리사이즈
                logger.warning("마커 검출 실패 → 강제 리사이즈로 fallback")
                image, resize_success = ensure_standard_size(image)
                debug_images['fallback_resize'] = image.copy()
                if not resize_success:
                    logger.error("강제 리사이즈도 실패")
        
        # 크기 확인 및 검증
        h, w = image.shape[:2]
        expected_w, expected_h = 3507, 2480
        if (w, h) != (expected_w, expected_h):
            logger.error(f"크기 불일치: {w}×{h} (기대값: {expected_w}×{expected_h})")
            # 최종 강제 리사이즈
            image, _ = ensure_standard_size(image)
            debug_images['final_resize'] = image.copy()
        
        logger.info(f"최종 이미지 크기: {image.shape}")

        template.image_instance_ops.reset_all_save_img()
        template.image_instance_ops.append_save_img(1, image)

        processed = template.image_instance_ops.apply_preprocessors(
            file_name, image, template
        )
        if processed is None:
            return None, "전처리 실패 (마커 인식 불가)"
        
        # 전처리 후 이미지 저장
        debug_images['06_preprocessed'] = processed.copy()
        
        # 전처리 후 크기 검증
        h, w = processed.shape[:2]
        if (w, h) != (expected_w, expected_h):
            logger.error(f"전처리 후 크기 불일치: {w}×{h} (기대값: {expected_w}×{expected_h})")
            # 크기가 틀리면 전처리 전 이미지로 대체
            processed = cv2.resize(image, (expected_w, expected_h), interpolation=cv2.INTER_AREA)
            debug_images['06_preprocessed'] = processed.copy()
            debug_images['preprocessed_resized'] = processed.copy()
        
        logger.info(f"전처리 후 크기: {processed.shape}")
        
        # 템플릿 버블 위치 표시 이미지 추가
        template_overlay = draw_template_bubbles_on_image(processed, template)
        debug_images['07_template_overlay'] = template_overlay

        response_dict, final_marked, multi_marked, multi_roll = (
            template.image_instance_ops.read_omr_response(
                template, image=processed, name=file_name, save_dir=None
            )
        )

        omr_response = get_concatenated_response(response_dict, template)
        omr_response = collections.OrderedDict(sorted(omr_response.items()))

        return {
            "file_name": file_name,
            "omr_response": omr_response,
            "multi_marked": bool(multi_marked),
            "final_marked": final_marked,
            "was_cropped": was_cropped,
            "debug_images": debug_images,
        }, None

    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)


def numpy_to_base64_jpeg(img, max_h=800):
    """numpy 이미지 -> base64 JPEG (미리보기용 리사이즈)"""
    h, w = img.shape[:2]
    if h > max_h:
        scale = max_h / h
        img = cv2.resize(img, (int(w * scale), max_h))
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("ascii")


def parse_answer_key_csv(csv_content):
    """답안 키 CSV 파싱 -> dict {question_label: answer}"""
    answer_key = {}
    try:
        lines = csv_content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                question = parts[0].strip().strip('"')
                answer = parts[1].strip().strip('"')
                answer_key[question] = answer
    except Exception as e:
        logger.error(f"Failed to parse answer key CSV: {e}")
    return answer_key


def normalize_answer(val):
    """
    답안 비교용 정규화: 선행 0 제거, 공백 제거.
    '08' -> '8', ' 03' -> '3', '8' -> '8', '' -> ''
    """
    val = val.strip()
    if not val:
        return val
    # 숫자로만 이루어진 경우 선행 0 제거 (예: '08' -> '8', '003' -> '3')
    # 단, '0' 자체는 '0'으로 유지
    try:
        return str(int(val))
    except ValueError:
        # 숫자가 아닌 경우 (A, B, AB 등) 그대로 반환
        return val


def score_response(omr_response, answer_key, correct_pts, incorrect_pts, unmarked_pts):
    """답안 키에 있는 문제만 채점. 선행 0 무시 ('08' == '8')."""
    total_score = 0.0
    correct_count = 0
    incorrect_count = 0
    unmarked_count = 0
    detail = {}

    for question, correct_answer in answer_key.items():
        marked = omr_response.get(question, "")
        marked_norm = normalize_answer(marked)
        answer_norm = normalize_answer(correct_answer)

        if not marked_norm:
            verdict = "unmarked"
            delta = unmarked_pts
            unmarked_count += 1
        elif marked_norm == answer_norm:
            verdict = "correct"
            delta = correct_pts
            correct_count += 1
        else:
            verdict = "incorrect"
            delta = incorrect_pts
            incorrect_count += 1

        total_score += delta
        detail[question] = verdict

    return total_score, correct_count, incorrect_count, unmarked_count, detail


# ────────────── HTML ──────────────

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OMR Quiz 채점 시스템</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --primary-light: #dbeafe;
            --success: #059669;
            --error: #dc2626;
            --warning: #d97706;
            --g50: #f9fafb; --g100: #f3f4f6; --g200: #e5e7eb;
            --g300: #d1d5db; --g500: #6b7280; --g700: #374151; --g900: #111827;
        }
        * { margin:0; padding:0; box-sizing:border-box; }
        body { font-family:'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif; background:var(--g50); color:var(--g900); min-height:100vh; }
        .container { max-width:1100px; margin:0 auto; padding:2rem 1.5rem; }

        header { text-align:center; margin-bottom:2rem; }
        header h1 { font-size:1.75rem; font-weight:700; margin-bottom:.4rem; }
        header p { color:var(--g500); font-size:.93rem; }

        .card { background:#fff; border-radius:12px; box-shadow:0 1px 3px rgba(0,0,0,.1); padding:1.75rem; margin-bottom:1.25rem; }
        .card h2 { font-size:1.05rem; font-weight:600; margin-bottom:.85rem; color:var(--g700); }
        .card h3 { font-size:.92rem; font-weight:600; margin:1rem 0 .6rem; color:var(--g700); }

        .step-badge { display:inline-block; width:24px; height:24px; border-radius:50%; background:var(--primary); color:#fff; text-align:center; line-height:24px; font-size:.75rem; font-weight:700; margin-right:.5rem; }

        /* Upload areas */
        .upload-area { border:2px dashed var(--g300); border-radius:10px; padding:2.5rem 1.5rem; text-align:center; cursor:pointer; transition:all .2s; background:var(--g50); }
        .upload-area:hover,.upload-area.dragover { border-color:var(--primary); background:var(--primary-light); }
        .upload-area svg { width:40px; height:40px; color:var(--g500); margin-bottom:.75rem; }
        .upload-area p { color:var(--g500); margin-bottom:.3rem; }
        .upload-area .hint { font-size:.78rem; color:var(--g500); }
        .upload-area .browse { color:var(--primary); font-weight:600; text-decoration:underline; }
        .upload-sm { padding:1.5rem 1rem; }
        .upload-sm svg { width:28px; height:28px; }

        /* File list */
        .file-list { margin-top:.75rem; max-height:180px; overflow-y:auto; }
        .file-item { display:flex; justify-content:space-between; align-items:center; padding:.4rem .65rem; border-radius:6px; font-size:.83rem; background:var(--g50); margin-bottom:.2rem; }
        .file-item .name { color:var(--g700); }
        .file-item .size { color:var(--g500); font-size:.78rem; }
        .file-item .remove { color:var(--error); cursor:pointer; font-weight:600; border:none; background:none; font-size:1.05rem; line-height:1; }

        /* Answer key info */
        .ak-info { display:flex; align-items:center; gap:.6rem; padding:.65rem .85rem; border-radius:8px; font-size:.83rem; margin-top:.6rem; }
        .ak-info.loaded { background:#ecfdf5; color:#065f46; }
        .ak-info.empty { background:var(--g100); color:var(--g500); }
        .csv-hint { font-size:.73rem; color:var(--g500); margin-top:.4rem; line-height:1.5; }
        .csv-hint code { background:var(--g100); padding:.08rem .3rem; border-radius:3px; font-size:.7rem; }

        /* Scoring params */
        .scoring-params { display:grid; grid-template-columns:repeat(3,1fr); gap:.65rem; margin-top:.6rem; }
        .scoring-params label { font-size:.78rem; color:var(--g500); display:block; margin-bottom:.2rem; }
        .scoring-params input { width:100%; padding:.45rem .65rem; border:1px solid var(--g300); border-radius:6px; font-size:.88rem; }
        .scoring-params input:focus { outline:none; border-color:var(--primary); box-shadow:0 0 0 2px var(--primary-light); }

        /* Buttons */
        .btn { display:inline-flex; align-items:center; justify-content:center; gap:.4rem; padding:.7rem 1.3rem; border-radius:8px; font-size:.93rem; font-weight:600; border:none; cursor:pointer; transition:all .15s; }
        .btn-primary { background:var(--primary); color:#fff; width:100%; margin-top:1.25rem; }
        .btn-primary:hover { background:var(--primary-dark); }
        .btn-primary:disabled { background:var(--g300); cursor:not-allowed; }
        .btn-success { background:var(--success); color:#fff; }
        .btn-success:hover { background:#047857; }
        .btn-outline { background:#fff; color:var(--primary); border:1.5px solid var(--primary); }
        .btn-outline:hover { background:var(--primary-light); }

        /* Progress */
        .progress-wrap { display:none; margin-top:1.25rem; }
        .progress-wrap.active { display:block; }
        .progress-bar-bg { width:100%; height:8px; background:var(--g200); border-radius:4px; overflow:hidden; }
        .progress-bar { height:100%; background:var(--primary); border-radius:4px; transition:width .3s; width:0%; }
        .progress-text { text-align:center; margin-top:.6rem; font-size:.88rem; color:var(--g500); }

        /* Results */
        .results-section { display:none; }
        .results-section.active { display:block; }

        .stats-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:.65rem; margin-bottom:1.25rem; }
        .stat-card { background:var(--g50); border-radius:8px; padding:.75rem; text-align:center; }
        .stat-card .number { font-size:1.3rem; font-weight:700; color:var(--primary); }
        .stat-card .label { font-size:.72rem; color:var(--g500); margin-top:.15rem; }
        .stat-card.error .number { color:var(--error); }
        .stat-card.warning .number { color:var(--warning); }
        .stat-card.success .number { color:var(--success); }

        /* Table */
        .table-wrap { overflow-x:auto; border:1px solid var(--g200); border-radius:8px; max-height:450px; }
        table { width:100%; border-collapse:collapse; font-size:.76rem; }
        th,td { padding:.4rem .5rem; text-align:left; border-bottom:1px solid var(--g200); white-space:nowrap; }
        th { background:var(--g50); font-weight:600; color:var(--g700); position:sticky; top:0; z-index:1; }
        tr:hover td { background:var(--primary-light); }
        tr.clickable { cursor:pointer; }
        td.cell-correct { background:#d1fae5; }
        td.cell-incorrect { background:#fee2e2; }

        .download-area { display:flex; gap:.75rem; margin-top:1.25rem; flex-wrap:wrap; }

        .info-badge { display:inline-block; padding:.12rem .45rem; border-radius:999px; font-size:.66rem; font-weight:600; }
        .badge-ok { background:#d1fae5; color:#065f46; }
        .badge-multi { background:#fef3c7; color:#92400e; }
        .badge-err { background:#fee2e2; color:#991b1b; }
        .badge-rotated { background:#dbeafe; color:#1e40af; }

        /* Debug modal */
        .modal-overlay { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,.7); z-index:100; align-items:center; justify-content:center; }
        .modal-overlay.active { display:flex; }
        .modal-content { background:#fff; border-radius:12px; max-width:90vw; max-height:90vh; overflow:auto; position:relative; }
        .modal-header { display:flex; justify-content:space-between; align-items:center; padding:1rem 1.25rem; border-bottom:1px solid var(--g200); position:sticky; top:0; background:#fff; z-index:2; border-radius:12px 12px 0 0; }
        .modal-header h3 { font-size:.95rem; font-weight:600; }
        .modal-close { background:none; border:none; font-size:1.4rem; cursor:pointer; color:var(--g500); padding:.2rem .5rem; }
        .modal-close:hover { color:var(--g900); }
        .modal-body { padding:1rem 1.25rem; }
        .modal-body img { max-width:100%; height:auto; }
        .modal-detail { margin-top:.75rem; }
        .modal-detail table { font-size:.8rem; }
        .modal-detail th { background:var(--g100); }

        .spinner { display:inline-block; width:18px; height:18px; border:2px solid rgba(255,255,255,.3); border-top:2px solid #fff; border-radius:50%; animation:spin .6s linear infinite; }
        @keyframes spin { to { transform:rotate(360deg); } }

        .hidden-input { display:none; }

        .toggle-row { display:flex; align-items:center; gap:.5rem; margin-bottom:.6rem; }
        .toggle-row input[type="checkbox"] { width:17px; height:17px; cursor:pointer; }
        .toggle-row label { font-size:.88rem; color:var(--g700); cursor:pointer; }

        .scoring-note { font-size:.8rem; color:var(--primary); background:var(--primary-light); padding:.5rem .75rem; border-radius:6px; margin-top:.5rem; }

        /* Guide accordion */
        .guide-card{padding:0;overflow:hidden}
        .guide-toggle{width:100%;display:flex;justify-content:space-between;align-items:center;padding:1rem 1.5rem;background:#fff;border:none;cursor:pointer;font-size:.95rem;font-weight:600;color:var(--g700);border-radius:12px}
        .guide-toggle:hover{background:var(--g50)}
        .guide-arrow{transition:transform .2s;font-size:.8rem;color:var(--g500)}
        .guide-body{display:none;padding:0 1.5rem 1.25rem;font-size:.82rem;color:var(--g500);line-height:1.7}
        .guide-body.open{display:block}
        .guide-arrow.open{transform:rotate(180deg)}
        .guide-body ol{padding-left:1.2rem;margin:.5rem 0}
        .guide-body li{margin-bottom:.3rem}
        .guide-body strong{color:var(--g700)}
        .guide-body code{background:var(--g100);padding:.1rem .3rem;border-radius:3px;font-size:.78rem}
        .ak-remove{background:none;border:none;color:var(--error);cursor:pointer;font-weight:600;font-size:.95rem;margin-left:auto}
        footer{text-align:center;padding:1.5rem 0 .5rem;font-size:.72rem;color:var(--g500);line-height:1.6}
        footer a{color:var(--primary);text-decoration:none} footer a:hover{text-decoration:underline}

        .ak-actions { margin-top:.5rem; }
        .template-dl { display:flex; align-items:center; gap:.5rem; margin-top:.35rem; flex-wrap:wrap; }
        .dl-link { font-size:.76rem; color:var(--primary); font-weight:600; text-decoration:none; padding:.2rem .55rem; border:1px solid var(--primary); border-radius:5px; transition:all .15s; }
        .dl-link:hover { background:var(--primary); color:#fff; }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>OMR Quiz 채점 시스템</h1>
        <p>답안 키와 OMR 스캔 이미지를 업로드하면 자동 인식 및 채점 결과를 CSV로 제공합니다</p>
    </header>

    <!-- Guide (folded by default) -->
    <div class="card guide-card">
        <button class="guide-toggle" onclick="document.getElementById('guideBody').classList.toggle('open');document.getElementById('guideArrow').classList.toggle('open');">
            <span>사용 방법 및 기능 안내</span>
            <span class="guide-arrow" id="guideArrow">&#9660;</span>
        </button>
        <div class="guide-body" id="guideBody">
            <ol>
                <li><strong>Step 1 - 답안 키 설정:</strong> 정답이 담긴 CSV 파일을 드래그하거나 클릭하여 업로드합니다. 빈 템플릿을 다운로드 받아 정답을 채워 사용하세요.</li>
                <li><strong>Step 2 - 스캔 이미지 업로드:</strong> 스캐너에서 출력된 OMR 답안지 이미지(JPG/PNG)를 여러 장 동시에 드래그하거나 선택합니다.</li>
                <li><strong>Step 3 - 결과 확인:</strong> 인식 결과가 테이블로 표시되며 CSV로 다운로드할 수 있습니다.</li>
            </ol>
            <p><strong>주요 기능:</strong></p>
            <ul style="padding-left:1.2rem;margin:.5rem 0;">
                <li><strong>용지 자동 크롭:</strong> 3개 코너 마커를 감지하여 OMR 용지 영역만 자동 정렬 및 크롭</li>
                <li><strong>컬러 드롭아웃:</strong> 빨간색 인쇄 템플릿이 마킹으로 오인식되지 않도록 컬러 잉크 제거</li>
                <li><strong>가로/세로 자동 판별:</strong> 세로 이미지는 양방향 회전 시도 후 최적 방향 자동 선택</li>
                <li><strong>선행 0 무시 채점:</strong> <code>08</code>과 <code>8</code>을 동일한 정답으로 인정</li>
                <li><strong>가변 문제 수:</strong> 답안 키 CSV에 있는 문제만 채점 (Q1~Q38이면 38문제만)</li>
                <li><strong>디버그 모드:</strong> 결과 행 클릭 시 OMR 버블 인식 결과 이미지 확인</li>
            </ul>
            <p><strong>답안 키 CSV 형식:</strong> 헤더 없이 한 줄에 <code>Q1,16</code> 형태. 콤마 구분.</p>
        </div>
    </div>

    <!-- STEP 1: Answer Key -->
    <div class="card" id="step1Card">
        <h2><span class="step-badge">1</span>답안 키 설정</h2>

        <div class="toggle-row">
            <input type="checkbox" id="enableScoring" checked>
            <label for="enableScoring">답안 키 CSV로 자동 채점</label>
        </div>

        <div id="scoringConfig">
            <div class="upload-area upload-sm" id="answerKeyArea">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
                </svg>
                <p>답안 키 CSV를 드래그하거나 <span class="browse">파일 선택</span></p>
            </div>
            <input type="file" id="answerKeyInput" class="hidden-input" accept=".csv">
            <div id="answerKeyInfo" class="ak-info empty"><span id="akInfoText">답안 키를 업로드해주세요</span></div>
            <div class="ak-actions">
                <p class="csv-hint">
                    CSV 형식: 한 줄에 <code>문제번호,정답</code> (예: <code>Q1,16</code>).
                    문제 수 자유 - Q1~Q38이면 38문제만 채점됩니다.
                </p>
                <div class="template-dl">
                    <span class="csv-hint">빈 템플릿:</span>
                    <a href="/template-csv/44" class="dl-link" download="answer_key_44.csv">Q1~Q44 (44문항)</a>
                    <a href="/template-csv/38" class="dl-link" download="answer_key_38.csv">Q1~Q38 (38문항)</a>
                    <a href="/template-csv/22" class="dl-link" download="answer_key_22.csv">Q1~Q22 (22문항)</a>
                </div>
            </div>

            <h3>배점 설정</h3>
            <div class="scoring-params">
                <div><label>정답 배점</label><input type="number" id="ptsCorrect" value="1" step="0.5"></div>
                <div><label>오답 배점</label><input type="number" id="ptsIncorrect" value="0" step="0.5"></div>
                <div><label>미기입 배점</label><input type="number" id="ptsUnmarked" value="0" step="0.5"></div>
            </div>
        </div>

        <div id="noScoringNote" class="scoring-note" style="display:none;">
            채점 없이 OMR 응답만 추출합니다. 모든 문제(Q1~Q44)의 인식 결과가 CSV에 포함됩니다.
        </div>
    </div>

    <!-- STEP 2: Images -->
    <div class="card" id="step2Card">
        <h2><span class="step-badge">2</span>답안지 이미지 업로드</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" id="uploadArea">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
                </svg>
                <p>스캔 이미지를 드래그하거나 <span class="browse">파일 선택</span></p>
                <p class="hint">JPG, PNG / 여러 장 동시 업로드 / 가로-세로 자동 판별</p>
            </div>
            <input type="file" id="fileInput" class="hidden-input" multiple accept=".jpg,.jpeg,.png">
            <div class="file-list" id="fileList"></div>

            <div class="toggle-row" style="margin-top:1rem;">
                <input type="checkbox" id="enableCrop" checked>
                <label for="enableCrop">용지 자동 크롭 (스캐너 여백 제거)</label>
            </div>
            <div class="toggle-row">
                <input type="checkbox" id="enableDebug" checked>
                <label for="enableDebug">디버그 모드 (인식 결과 이미지 확인)</label>
            </div>

            <button type="submit" class="btn btn-primary" id="submitBtn" disabled>채점 시작</button>
        </form>

        <div class="progress-wrap" id="progressWrap">
            <div class="progress-bar-bg"><div class="progress-bar" id="progressBar"></div></div>
            <p class="progress-text" id="progressText">처리 중...</p>
        </div>
    </div>

    <!-- STEP 3: Results -->
    <div class="results-section" id="resultsSection">
        <div class="card">
            <h2><span class="step-badge">3</span>채점 결과</h2>
            <div class="stats-grid" id="statsGrid"></div>
            <p id="debugHint" style="font-size:.8rem; color:var(--g500); margin-bottom:.5rem; display:none;">
                행을 클릭하면 OMR 인식 결과 이미지를 확인할 수 있습니다.
            </p>
            <div class="table-wrap">
                <table id="resultsTable">
                    <thead id="tableHead"></thead>
                    <tbody id="tableBody"></tbody>
                </table>
            </div>
            <div class="download-area">
                <a id="downloadCsvBtn" class="btn btn-success" href="#" download>CSV 다운로드</a>
                <button class="btn btn-outline" onclick="resetForm()">새로 채점하기</button>
            </div>
        </div>
    </div>
    <footer>
        Web interface by <strong>june-oh</strong> |
        Based on <a href="https://github.com/Udayraj123/OMRChecker" target="_blank">OMRChecker</a> by Udayraj Deshmukh |
        <a href="https://github.com/Udayraj123/OMRChecker/blob/master/LICENSE" target="_blank">MIT License</a>
    </footer>
</div>

<!-- Debug Image Modal -->
<div class="modal-overlay" id="debugModal">
    <div class="modal-content">
        <div class="modal-header">
            <h3 id="modalTitle">인식 결과</h3>
            <button class="modal-close" onclick="closeModal()">&times;</button>
        </div>
        <div class="modal-body">
            <div id="modalImage" style="max-height: 70vh; overflow-y: auto;"></div>
            <div class="modal-detail" id="modalDetail"></div>
        </div>
    </div>
</div>

<script>
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const submitBtn = document.getElementById('submitBtn');
const uploadForm = document.getElementById('uploadForm');
const progressWrap = document.getElementById('progressWrap');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const resultsSection = document.getElementById('resultsSection');
const enableScoring = document.getElementById('enableScoring');
const scoringConfig = document.getElementById('scoringConfig');
const noScoringNote = document.getElementById('noScoringNote');
const answerKeyArea = document.getElementById('answerKeyArea');
const answerKeyInput = document.getElementById('answerKeyInput');
const answerKeyInfo = document.getElementById('answerKeyInfo');
const enableDebug = document.getElementById('enableDebug');

let selectedFiles = [];
let answerKeyFile = null;
let lastResults = null;  // store for debug modal

// Toggle scoring
enableScoring.addEventListener('change', () => {
    scoringConfig.style.display = enableScoring.checked ? 'block' : 'none';
    noScoringNote.style.display = enableScoring.checked ? 'none' : 'block';
});

// Answer key upload (click + drag & drop)
answerKeyArea.addEventListener('click', () => answerKeyInput.click());
answerKeyArea.addEventListener('dragover', e => { e.preventDefault(); answerKeyArea.classList.add('dragover'); });
answerKeyArea.addEventListener('dragleave', () => answerKeyArea.classList.remove('dragover'));
answerKeyArea.addEventListener('drop', e => {
    e.preventDefault();
    answerKeyArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        loadAnswerKey(e.dataTransfer.files[0]);
    }
});
answerKeyInput.addEventListener('change', () => {
    if (answerKeyInput.files.length > 0) loadAnswerKey(answerKeyInput.files[0]);
});

function loadAnswerKey(file) {
    answerKeyFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        const lines = e.target.result.trim().split('\n').filter(l => l.trim());
        answerKeyInfo.className = 'ak-info loaded';
        answerKeyInfo.innerHTML = `<span>${answerKeyFile.name} — ${lines.length}개 문제 답안 로드 완료</span><button class="ak-remove" onclick="removeAnswerKey(event)" title="제거">&times;</button>`;
    };
    reader.readAsText(answerKeyFile);
}
function removeAnswerKey(e) {
    e.stopPropagation();
    answerKeyFile = null; answerKeyInput.value = '';
    answerKeyInfo.className = 'ak-info empty';
    answerKeyInfo.innerHTML = '<span>답안 키를 업로드해주세요</span>';
}

// Image upload
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => { e.preventDefault(); uploadArea.classList.remove('dragover'); addFiles(e.dataTransfer.files); });
fileInput.addEventListener('change', () => addFiles(fileInput.files));

function addFiles(files) {
    for (const f of files) {
        if (!selectedFiles.find(s => s.name === f.name && s.size === f.size)) selectedFiles.push(f);
    }
    renderFileList();
}
function removeFile(i) { selectedFiles.splice(i, 1); renderFileList(); }
function formatSize(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
    return (b/1048576).toFixed(1) + ' MB';
}
function renderFileList() {
    fileList.innerHTML = selectedFiles.map((f,i) =>
        `<div class="file-item"><span class="name">${f.name}</span><span class="size">${formatSize(f.size)}</span><button class="remove" onclick="removeFile(${i})">&times;</button></div>`
    ).join('');
    submitBtn.disabled = selectedFiles.length === 0;
}

// Submit
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (selectedFiles.length === 0) return;

    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> 처리 중...';
    progressWrap.classList.add('active');
    progressBar.style.width = '10%';
    progressText.textContent = `${selectedFiles.length}장 업로드 중...`;

    const formData = new FormData();
    selectedFiles.forEach(f => formData.append('files', f));

    if (enableScoring.checked && answerKeyFile) {
        formData.append('answer_key', answerKeyFile);
        formData.append('pts_correct', document.getElementById('ptsCorrect').value);
        formData.append('pts_incorrect', document.getElementById('ptsIncorrect').value);
        formData.append('pts_unmarked', document.getElementById('ptsUnmarked').value);
    }

    if (document.getElementById('enableCrop').checked) formData.append('crop', '1');
    if (enableDebug.checked) formData.append('debug', '1');

    try {
        progressBar.style.width = '30%';
        progressText.textContent = 'OMR 인식 처리 중...';
        const response = await fetch('/process', { method: 'POST', body: formData });
        progressBar.style.width = '90%';

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || '서버 오류');
        }

        const data = await response.json();
        lastResults = data;
        progressBar.style.width = '100%';
        progressText.textContent = '완료!';

        setTimeout(() => {
            displayResults(data);
            progressWrap.classList.remove('active');
            submitBtn.innerHTML = '채점 시작';
            submitBtn.disabled = false;
        }, 400);
    } catch (err) {
        progressWrap.classList.remove('active');
        submitBtn.innerHTML = '채점 시작';
        submitBtn.disabled = false;
        alert('오류: ' + err.message);
    }
});

function displayResults(data) {
    const { results, errors, summary, csv_download_id, has_scoring, answer_key, has_debug } = data;
    const columns = data.columns;

    // Stats
    let statsHtml = `
        <div class="stat-card success"><div class="number">${summary.total_processed}</div><div class="label">처리 완료</div></div>
        <div class="stat-card"><div class="number">${summary.total_uploaded}</div><div class="label">업로드</div></div>
        <div class="stat-card warning"><div class="number">${summary.multi_marked}</div><div class="label">다중 마킹</div></div>
        <div class="stat-card error"><div class="number">${summary.errors}</div><div class="label">인식 오류</div></div>`;

    if (has_scoring && summary.avg_score !== undefined) {
        statsHtml += `
        <div class="stat-card success"><div class="number">${summary.avg_score}</div><div class="label">평균 점수</div></div>
        <div class="stat-card"><div class="number">${summary.max_score}</div><div class="label">최고 점수</div></div>
        <div class="stat-card"><div class="number">${summary.min_score}</div><div class="label">최저 점수</div></div>
        <div class="stat-card"><div class="number">${summary.total_questions}</div><div class="label">채점 문제 수</div></div>`;
    }
    document.getElementById('statsGrid').innerHTML = statsHtml;

    // Debug hint
    document.getElementById('debugHint').style.display = has_debug ? 'block' : 'none';

    // Table
    const tableHead = document.getElementById('tableHead');
    const tableBody = document.getElementById('tableBody');
    tableHead.innerHTML = '<tr>' + columns.map(c => `<th>${c}</th>`).join('') + '<th>상태</th></tr>';
    tableBody.innerHTML = '';

    results.forEach((r, idx) => {
        const row = document.createElement('tr');
        if (has_debug) row.classList.add('clickable');
        if (has_debug) row.addEventListener('click', () => openModal(idx));

        columns.forEach(c => {
            const td = document.createElement('td');
            td.textContent = r[c] !== undefined && r[c] !== null ? r[c] : '';
            if (has_scoring && answer_key && answer_key[c] !== undefined) {
                const v = String(r[c] || '').trim();
                const a = String(answer_key[c] || '').trim();
                // 선행 0 무시 비교: '08' == '8'
                const vn = /^\d+$/.test(v) ? String(parseInt(v,10)) : v;
                const an = /^\d+$/.test(a) ? String(parseInt(a,10)) : a;
                if (vn && vn === an) td.classList.add('cell-correct');
                else if (vn) td.classList.add('cell-incorrect');
            }
            row.appendChild(td);
        });

        const td = document.createElement('td');
        let badges = '<span class="info-badge badge-ok">OK</span>';
        if (r._multi_marked) badges = '<span class="info-badge badge-multi">다중마킹</span>';
        if (r._rotated) badges += ' <span class="info-badge badge-rotated">회전</span>';
        if (r._cropped) badges += ' <span class="info-badge badge-rotated">크롭</span>';
        td.innerHTML = badges;
        row.appendChild(td);
        tableBody.appendChild(row);
    });

    errors.forEach(err => {
        const row = document.createElement('tr');
        const td1 = document.createElement('td'); td1.textContent = err.file_name; row.appendChild(td1);
        for (let i = 1; i < columns.length; i++) { const td = document.createElement('td'); td.textContent = '-'; row.appendChild(td); }
        const td = document.createElement('td');
        td.innerHTML = `<span class="info-badge badge-err" title="${err.error}">오류</span>`;
        row.appendChild(td);
        tableBody.appendChild(row);
    });

    document.getElementById('downloadCsvBtn').href = `/download/${csv_download_id}`;
    resultsSection.classList.add('active');
}

// Debug modal
function openModal(idx) {
    if (!lastResults || !lastResults.results[idx]) return;
    const r = lastResults.results[idx];
    const debugImg = r._debug_image;
    const debugImages = r._debug_images;
    
    if (!debugImg && !debugImages) return;

    document.getElementById('modalTitle').textContent = `인식 결과: ${r.file_id}`;
    
    // 여러 디버그 이미지가 있으면 표시
    let imageHtml = '';
    if (debugImages && Object.keys(debugImages).length > 0) {
        const stages = [
            {key: '01_original', label: '1. 원본 이미지'},
            {key: '02_color_dropout', label: '2. 컬러 드롭아웃 (흑백)'},
            {key: '03_rotated', label: '3. 회전 보정'},
            {key: '04_markers_detected', label: '4. 마커 검출'},
            {key: '05_warped', label: '5. Affine 변환 (크롭)'},
            {key: 'fallback_resize', label: '⚠️ Fallback 리사이즈 (마커 없음)'},
            {key: '06_preprocessed', label: '6. 전처리 완료'},
            {key: '07_template_overlay', label: '7. 템플릿 버블 위치'},
            {key: 'final_resize', label: '⚠️ 최종 크기 보정'}
        ];
        
        stages.forEach(stage => {
            if (debugImages[stage.key]) {
                imageHtml += `
                    <div style="margin-bottom: 20px;">
                        <h4 style="margin: 10px 0; color: #333; font-size: 14px;">${stage.label}</h4>
                        <img src="data:image/jpeg;base64,${debugImages[stage.key]}" 
                             style="max-width: 100%; border: 1px solid #ddd;">
                        <p style="font-size: 11px; color: #666; margin-top: 4px;">키: ${stage.key}</p>
                    </div>`;
            }
        });
        
        // 최종 결과 (마킹 표시)
        if (debugImg) {
            imageHtml += `
                <div style="margin-bottom: 20px;">
                    <h4 style="margin: 10px 0; color: #333; font-size: 14px;">8. 최종 인식 결과 (채점)</h4>
                    <img src="data:image/jpeg;base64,${debugImg}" 
                         style="max-width: 100%; border: 1px solid #ddd;">
                </div>`;
        }
    } else if (debugImg) {
        // 기존 방식 (호환성)
        imageHtml = `<img id="modalImage" src="data:image/jpeg;base64,${debugImg}" style="max-width: 100%;">`;
    }
    
    document.getElementById('modalImage').innerHTML = imageHtml;

    // Detail table
    let detailHtml = '';
    if (lastResults.has_scoring && lastResults.answer_key && r.score !== undefined) {
        detailHtml = `<p style="font-size:.85rem; margin-bottom:.5rem;"><strong>점수: ${r.score}</strong> (정답 ${r._correct || 0} / 오답 ${r._incorrect || 0} / 미기입 ${r._unmarked || 0})</p>`;
    }
    document.getElementById('modalDetail').innerHTML = detailHtml;

    document.getElementById('debugModal').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    document.getElementById('debugModal').classList.remove('active');
    document.body.style.overflow = '';
}
document.getElementById('debugModal').addEventListener('click', (e) => {
    if (e.target === document.getElementById('debugModal')) closeModal();
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });

function resetForm() {
    selectedFiles = [];
    answerKeyFile = null;
    lastResults = null;
    renderFileList();
    answerKeyInput.value = '';
    answerKeyInfo.className = 'ak-info empty';
    answerKeyInfo.innerHTML = '<span>답안 키를 업로드해주세요</span>';
    resultsSection.classList.remove('active');
    progressWrap.classList.remove('active');
}
</script>
</body>
</html>
"""

# ── 임시 CSV 저장소 ──
_csv_store = {}


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/process", methods=["POST"])
def process_omr():
    if "files" not in request.files:
        return jsonify({"error": "파일이 없습니다"}), 400

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"error": "파일을 선택해주세요"}), 400

    try:
        template = get_template()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    # ── 답안 키 ──
    answer_key = None
    correct_pts = 1.0
    incorrect_pts = 0.0
    unmarked_pts = 0.0
    has_scoring = False

    if "answer_key" in request.files:
        ak_file = request.files["answer_key"]
        if ak_file.filename:
            try:
                ak_content = ak_file.read().decode("utf-8-sig")
                answer_key = parse_answer_key_csv(ak_content)
                if answer_key:
                    has_scoring = True
                    correct_pts = float(request.form.get("pts_correct", 1))
                    incorrect_pts = float(request.form.get("pts_incorrect", 0))
                    unmarked_pts = float(request.form.get("pts_unmarked", 0))
                    logger.info(
                        f"Answer key: {len(answer_key)} questions, "
                        f"+{correct_pts}/{incorrect_pts}/{unmarked_pts}"
                    )
            except Exception as e:
                logger.error(f"Failed to parse answer key: {e}")

    # ── 디버그 모드 ──
    has_debug = request.form.get("debug") == "1"

    results = []
    errors = []
    rotated_count = 0

    output_columns = template.output_columns  # ["Num", "Q1", ..., "Q44"]

    # CSV 컬럼 구성
    if has_scoring:
        csv_columns = ["file_id", "score"] + output_columns
    else:
        csv_columns = ["file_id"] + output_columns

    for file in files:
        if not file.filename:
            continue
        file_name = file.filename

        try:
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)

            # 컬러로 읽어서 Red 채널 드롭아웃 적용
            # (빨간 인쇄 템플릿이 흑백 변환 시 검정으로 오인식되는 문제 해결)
            color_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if color_image is None:
                errors.append({"file_name": file_name, "error": "이미지를 읽을 수 없습니다"})
                continue

            # 컬러 드롭아웃: max(B,G,R) → 모든 컬러 잉크 밝게, 검정 마킹만 어둡게
            b, g, r = cv2.split(color_image)
            image = np.maximum(np.maximum(b, g), r)

            # 크롭 활성화 시: marker_align_and_crop이 회전+정렬+크롭+Red드롭아웃 일괄 수행
            # 크롭 비활성화 시: 기존 auto_rotate만 수행 (이미 Red 채널 적용됨)
            enable_crop = request.form.get("crop") == "1"

            was_rotated = False
            if not enable_crop:
                image, was_rotated = auto_rotate_image(image, template)
                if was_rotated:
                    rotated_count += 1

            result, error_msg = process_single_image(
                template, image, file_name,
                enable_crop=enable_crop,
                color_image=color_image,
            )
            if error_msg:
                errors.append({"file_name": file_name, "error": error_msg})
                continue

            row = {"file_id": file_name}
            for col in output_columns:
                row[col] = result["omr_response"].get(col, "")
            row["_multi_marked"] = result["multi_marked"]
            row["_rotated"] = was_rotated
            row["_cropped"] = result.get("was_cropped", False)

            # 채점 (답안 키에 있는 문제만)
            if has_scoring and answer_key:
                total_score, correct_cnt, incorrect_cnt, unmarked_cnt, detail = (
                    score_response(
                        result["omr_response"],
                        answer_key,
                        correct_pts,
                        incorrect_pts,
                        unmarked_pts,
                    )
                )
                row["score"] = round(total_score, 2)
                row["_correct"] = correct_cnt
                row["_incorrect"] = incorrect_cnt
                row["_unmarked"] = unmarked_cnt

            # 디버그 이미지들
            if has_debug and result.get("debug_images"):
                row["_debug_images"] = {}
                for key, img in result["debug_images"].items():
                    if img is not None:
                        row["_debug_images"][key] = numpy_to_base64_jpeg(img, max_h=600)
            
            # 최종 마킹 이미지 (기존 호환성 유지)
            if has_debug and result["final_marked"] is not None:
                row["_debug_image"] = numpy_to_base64_jpeg(result["final_marked"])

            results.append(row)

        except Exception as e:
            logger.error(f"Unexpected error for {file_name}: {e}")
            errors.append({"file_name": file_name, "error": str(e)})

    # CSV
    csv_id = str(uuid.uuid4())
    if results:
        df = pd.DataFrame(results)
        csv_df = df[[c for c in csv_columns if c in df.columns]].copy()
        csv_buffer = StringIO()
        csv_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        _csv_store[csv_id] = csv_buffer.getvalue()
    else:
        csv_buffer = StringIO()
        pd.DataFrame(columns=csv_columns).to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        _csv_store[csv_id] = csv_buffer.getvalue()

    summary = {
        "total_uploaded": len(files),
        "total_processed": len(results),
        "multi_marked": sum(1 for r in results if r.get("_multi_marked")),
        "errors": len(errors),
        "rotated": rotated_count,
    }

    if has_scoring and results:
        scores = [r.get("score", 0) for r in results if "score" in r]
        if scores:
            summary["avg_score"] = round(sum(scores) / len(scores), 2)
            summary["max_score"] = round(max(scores), 2)
            summary["min_score"] = round(min(scores), 2)
            summary["total_questions"] = len(answer_key)

    return jsonify({
        "results": results,
        "errors": errors,
        "columns": csv_columns,
        "summary": summary,
        "csv_download_id": csv_id,
        "has_scoring": has_scoring,
        "answer_key": answer_key if has_scoring else None,
        "has_debug": has_debug,
    })


@app.route("/template-csv/<int:count>")
def download_template_csv(count):
    """빈 답안 키 템플릿 CSV 다운로드 (Q1~Qn)"""
    count = max(1, min(count, 44))
    lines = [f"Q{i}," for i in range(1, count + 1)]
    content = "\n".join(lines) + "\n"
    buffer = BytesIO(content.encode("utf-8-sig"))
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"answer_key_{count}.csv",
    )


@app.route("/download/<csv_id>")
def download_csv(csv_id):
    csv_content = _csv_store.get(csv_id)
    if csv_content is None:
        return "CSV를 찾을 수 없습니다. 다시 채점해주세요.", 404

    buffer = BytesIO(csv_content.encode("utf-8-sig"))
    return send_file(
        buffer,
        mimetype="text/csv",
        as_attachment=True,
        download_name="omr_results.csv",
    )


if __name__ == "__main__":
    try:
        get_template()
        logger.info("Template loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load template: {e}")

    print("\n" + "=" * 50)
    print("  OMR Quiz 채점 웹 서버")
    print("  http://localhost:5000 에서 접속하세요")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
