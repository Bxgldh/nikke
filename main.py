"""
sum10_overlay_optimized.py

优化版本（单文件） — 在 Windows 上支持 click-through overlay（若运行在其它平台则退回普通窗口）

关键优化：
 - 矩形查找：由 O(R^2 * C^2) -> O(R^2 * C)（按上下行压缩列并用 hashmap 找子数组和为 target）
 - OCR：仅对检测为“变化”的单元格执行 OCR（用 per-cell fast checksum / pixel-diff）
 - 并行 OCR：使用 ThreadPoolExecutor 提升多核处理吞吐
 - 减少图像合成与重绘，仅在 overlay 内容变化时合成
 - 小睡眠分片以保持响应性

依赖:
 pip install mss opencv-python numpy pytesseract pillow

注意:
 - 需要 tesseract 可用（若不在 PATH，可在脚本中配置 pytesseract.pytesseract.tesseract_cmd）
 - 若需要把 overlay 完全透明且不显示 preview，可把 composite 部分注释掉（见代码注释）
"""

import time, json, sys, os
from pathlib import Path
import numpy as np
import cv2
import mss
from PIL import Image
# ===== resource helper (放在文件开头) =====
import sys, os

def resource_path(rel_path):
    """
    Return absolute path to a resource, working for both:
    - development (return path relative to the .py file)
    - PyInstaller onefile (files unpacked to sys._MEIPASS)
    """
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)
# =============================================
import pytesseract
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import zlib

# ------------------ TUNABLE PARAMETERS ------------------
OCR_SCALE = 2                     # OCR 放大倍数
TESS_CONFIG = r'--psm 10 -c tessedit_char_whitelist=0123456789'
CAPTURE_INTERVAL = 0.45           # 基本循环间隔 (s)。减小更实时但占更多 CPU
CELL_CHANGE_THRESHOLD = 0.02      # 单元格变化阈值（归一化平均差）0~1，调整灵敏度
MAX_OCR_WORKERS = 4               # 并行 OCR 最大线程数（与 CPU 核、Tesseract 进程数相关）
TARGET_SUM = 10
MATRIX_JSON = Path("matrix.json")
# If tesseract not in PATH, set e.g.
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# -------------------------------------------------------

# ===== resource helper (放在文件开头) =====
import sys, os

def resource_path(rel_path):
    """
    Return absolute path to a resource, working for both:
    - development (return path relative to the .py file)
    - PyInstaller onefile (files unpacked to sys._MEIPASS)
    """
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)
# =============================================

IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import ctypes
    user32 = ctypes.windll.user32
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_TOPMOST = 0x00000008
    SetWindowLong = user32.SetWindowLongW
    GetWindowLong = user32.GetWindowLongW
    SetWindowPos = user32.SetWindowPos
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    HWND_TOPMOST = -1
    FindWindow = user32.FindWindowW

# ------------------ util / OCR / image helper ------------------
def grab_region(monitor):
    with mss.mss() as sct:
        img = sct.grab(monitor)
        arr = np.array(img)  # BGRA
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return bgr

def preprocess_for_ocr(cell_img):
    # fast preprocessing for OCR: grayscale -> resize -> blur -> otsu
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h,w) <= 0:
        return gray
    gray = cv2.resize(gray, (int(w*OCR_SCALE), int(h*OCR_SCALE)), interpolation=cv2.INTER_LINEAR)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) < 127:
        th = 255 - th
    return th

def ocr_cell_image(cell_img):
    # receives BGR cell image
    th = preprocess_for_ocr(cell_img)
    pil = Image.fromarray(th)
    txt = pytesseract.image_to_string(pil, config=TESS_CONFIG)
    txt = txt.strip()
    if not txt:
        return 0
    digits = ''.join(ch for ch in txt if ch.isdigit())
    if not digits:
        return 0
    return int(digits[0])

# fast fingerprint for a small cell image: downsample -> mean/std or crc32 to detect change
def cell_fingerprint(cell_img):
    # downscale to small fixed size, convert grayscale, return crc32 of bytes
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (12,12), interpolation=cv2.INTER_AREA)
    # normalize and quantize to bytes to reduce noise sensitivity
    # use uint8 bytes directly
    return zlib.crc32(small.tobytes())

# ------------------ grid utilities ------------------
def split_grid_regular(bgr_img, rows, cols):
    H, W = bgr_img.shape[:2]
    cell_h = H / rows
    cell_w = W / cols
    boxes = []
    for r in range(rows):
        for c in range(cols):
            top = int(round(r * cell_h + 1))
            left = int(round(c * cell_w + 1))
            bottom = int(round((r+1) * cell_h - 1))
            right = int(round((c+1) * cell_w - 1))
            boxes.append((top, left, bottom, right))
    return boxes

def read_matrix_ocr_partial(img, rows, cols, boxes, prev_fps, executor):
    """
    只对发生变化的 cell 做 OCR（并行）。
    返回 matrix, new_fps
    """
    matrix = [[0]*cols for _ in range(rows)]
    new_fps = [None] * (rows*cols)
    tasks = {}
    # first compute fingerprints and decide which cells changed
    for idx, box in enumerate(boxes):
        top,left,bottom,right = box
        roi = img[top:bottom, left:right]
        fp = cell_fingerprint(roi)
        new_fps[idx] = fp
        if prev_fps is None or prev_fps[idx] != fp:
            # changed -> submit OCR
            future = executor.submit(ocr_cell_image, roi)
            tasks[future] = idx
        else:
            # unchanged -> keep previous value (we'll fill after)
            pass
    # collect results
    changed_indices = set()
    for fut in as_completed(tasks):
        idx = tasks[fut]
        try:
            val = fut.result(timeout=5)
        except Exception:
            val = 0
        r = idx // cols; c = idx % cols
        matrix[r][c] = int(val)
        changed_indices.add(idx)
    # fill unchanged cells with sentinel -1 to indicate "no new OCR"
    # caller may choose to reuse previous matrix values
    return matrix, new_fps, changed_indices

# ------------------ rectangle search (optimized O(R^2 * C)) ------------------
def find_all_sum_rects_hashmap(mat, target=TARGET_SUM):
    """
    Find all axis-aligned submatrices whose sum == target.
    Complexity: O(R^2 * C) using hashmap over prefix sums across columns.
    Returns list of dict {r1,c1,r2,c2,area}
    """
    R = len(mat); C = len(mat[0])
    rects = []
    # for each pair of top r1 and bottom r2, collapse rows into 1D array of column sums
    for r1 in range(R):
        col_sums = [0]*C
        for r2 in range(r1, R):
            # add row r2 to col_sums
            row = mat[r2]
            for c in range(C):
                col_sums[c] += row[c]
            # now find subarrays in col_sums with sum == target
            # use prefix sum and hashmap mapping prefix_sum -> list of indices (we store first occurrence index)
            prefix = 0
            seen = {0: [-1]}  # prefix_sum -> list of indices where it occurred (starting with index -1)
            for c_idx in range(C):
                prefix += col_sums[c_idx]
                need = prefix - target
                if need in seen:
                    # for each start index s in seen[need], subarray (s+1 ... c_idx) sums to target
                    for start_idx in seen[need]:
                        c1 = start_idx + 1
                        c2 = c_idx
                        rects.append({'r1': r1, 'c1': c1, 'r2': r2, 'c2': c2, 'area': (r2-r1+1)*(c2-c1+1)})
                # append current index to seen[prefix]
                if prefix in seen:
                    seen[prefix].append(c_idx)
                else:
                    seen[prefix] = [c_idx]
    return rects

def greedy_select_nonoverlap(rects, rows, cols, strategy='areaAsc'):
    chosen = []
    occupied = [[False]*cols for _ in range(rows)]
    if strategy == 'areaAsc':
        rects.sort(key=lambda x: x['area'])
    else:
        rects.sort(key=lambda x: -x['area'])
    for r in rects:
        overlap = False
        for rr in range(r['r1'], r['r2']+1):
            for cc in range(r['c1'], r['c2']+1):
                if occupied[rr][cc]:
                    overlap = True; break
            if overlap:
                break
        if not overlap:
            chosen.append(r)
            for rr in range(r['r1'], r['r2']+1):
                for cc in range(r['c1'], r['c2']+1):
                    occupied[rr][cc] = True
    return chosen

# ------------------ overlay drawing ------------------
def draw_overlays_canvas(img_size, rects, rows, cols):
    H, W = img_size
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    cell_h = H / rows; cell_w = W / cols
    palette = [
        (230,25,75),(60,180,75),(255,225,25),(0,130,200),
        (245,130,48),(145,30,180),(70,240,240),(240,50,230),
        (210,245,60),(250,190,190)]
    for i, r in enumerate(rects):
        color = palette[i % len(palette)]
        left = int(round(r['c1'] * cell_w))
        top  = int(round(r['r1'] * cell_h))
        right = int(round((r['c2']+1) * cell_w)) - 1
        bottom = int(round((r['r2']+1) * cell_h)) - 1
        cv2.rectangle(canvas, (left, top), (right, bottom), (color[2], color[1], color[0], 220), 3, cv2.LINE_AA)
        label = f"{r['r1']},{r['c1']}-{r['r2']},{r['c2']}"
        txt_sz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bx1 = left+4; by1 = top+18 - txt_sz[1]; bx2 = bx1 + txt_sz[0] + 6; by2 = top+18 + 4
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0,0,0,120), -1)
        cv2.putText(canvas, label, (left+6, top+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return canvas

# ------------------ interactive selection (two clicks) ------------------
def select_region_interactive():
    with mss.mss() as sct:
        mon = sct.monitors[1]
        sct_img = sct.grab(mon)
        arr = np.array(sct_img)
        img = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    clone = img.copy()
    pts = []
    def mouse_cb(event, x, y, flags, param):
        nonlocal pts, clone, img
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y))
        elif event == cv2.EVENT_MOUSEMOVE and len(pts)==1:
            tmp = clone.copy()
            cv2.rectangle(tmp, pts[0], (x,y), (0,255,0), 2)
            cv2.imshow("Select region - click two points (L)", tmp)
    cv2.namedWindow("Select region - click two points (L)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("Select region - click two points (L)", mouse_cb)
    cv2.imshow("Select region - click two points (L)", img)
    print("请在弹出窗口里左键点击两次选择区域（起点、终点）。Esc 取消。")
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            return None
        if len(pts) >= 2:
            break
    cv2.destroyAllWindows()
    p1 = pts[0]; p2 = pts[1]
    left = min(p1[0], p2[0]); top = min(p1[1], p2[1])
    right = max(p1[0], p2[0]); bottom = max(p1[1], p2[1])
    w = right - left + 1; h = bottom - top + 1
    with mss.mss() as sct:
        mon = sct.monitors[1]
    monitor = {'left': int(mon['left'] + left), 'top': int(mon['top'] + top), 'width': int(w), 'height': int(h)}
    print("选择区域：", monitor)
    return monitor

# ------------------ Windows click-through helper ------------------
def make_window_clickthrough_by_title(window_name):
    if not IS_WINDOWS:
        return False
    hwnd = FindWindow(None, window_name)
    if hwnd == 0:
        return False
    ex = GetWindowLong(hwnd, GWL_EXSTYLE)
    ex |= (WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST)
    SetWindowLong(hwnd, GWL_EXSTYLE, ex)
    SetWindowPos(hwnd, HWND_TOPMOST, 0,0,0,0, SWP_NOMOVE | SWP_NOSIZE)
    return True

# ------------------ main optimized loop ------------------
def main():
    print("=== sum10_overlay_optimized ===")
    region = select_region_interactive()
    if region is None:
        print("未选区，退出。")
        return
    try:
        rows = int(input("请输入行数 (rows)，例如 6: ").strip())
        cols = int(input("请输入列数 (cols)，例如 8: ").strip())
    except Exception:
        print("行列输入错误，退出."); return
    print("region:", region, "grid:", rows, "x", cols)

    boxes = None
    last_fps = None
    last_matrix = None
    last_chosen = []
    last_canvas = None

    overlay_name = "SUM10_OPT_OVERLAY"
    cv2.namedWindow(overlay_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(overlay_name, region['width'], region['height'])
    cv2.moveWindow(overlay_name, region['left'], region['top'])
    # attempt click-through
    if IS_WINDOWS:
        cv2.waitKey(50)
        ok = make_window_clickthrough_by_title(overlay_name)
        print("click-through set:", ok)

    executor = ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS)

    running = True
    single_step = False
    show_overlay = True

    try:
        while running:
            t0 = time.time()
            img = grab_region(region)  # capture region
            if boxes is None:
                boxes = split_grid_regular(img, rows, cols)
            # fast per-cell fingerprint and partial OCR
            partial_matrix, new_fps, changed_indices = read_matrix_ocr_partial(img, rows, cols, boxes, last_fps, executor)
            # build final matrix by merging unchanged values from last_matrix
            if last_matrix is None:
                # fill all from partial_matrix (partial stores values only for changed cells)
                # but partial_matrix contains OCR only for changed indices, so fill missing by running OCR synchronously (rare)
                # For first frame, we should OCR all cells (submit tasks)
                # simplified approach: if last_matrix is None, run OCR on all cells in parallel
                all_tasks = {}
                for idx, box in enumerate(boxes):
                    top,left,bottom,right = box
                    roi = img[top:bottom, left:right]
                    all_tasks[executor.submit(ocr_cell_image, roi)] = idx
                full_mat = [[0]*cols for _ in range(rows)]
                for fut in as_completed(all_tasks):
                    idx = all_tasks[fut]
                    try:
                        v = fut.result(timeout=5)
                    except Exception:
                        v = 0
                    r = idx // cols; c = idx % cols
                    full_mat[r][c] = int(v)
                matrix = full_mat
                last_matrix = matrix
                last_fps = new_fps
            else:
                # reuse last_matrix where no change
                matrix = [row[:] for row in last_matrix]
                for idx in changed_indices:
                    r = idx // cols; c = idx % cols
                    matrix[r][c] = partial_matrix[r][c]
                last_matrix = matrix
                last_fps = new_fps

            # write debug JSON (optional)
            try:
                MATRIX_JSON.write_text(json.dumps({'rows':rows,'cols':cols,'matrix':matrix,'timestamp':time.time()}, ensure_ascii=False, indent=2))
            except Exception:
                pass

            # decide whether to recompute rectangles:
            changed = (last_chosen is None) or (not matrices_equal(matrix, last_matrix)) or single_step
            # Actually matrix just updated; we recompute every time matrix changed or single_step
            # Use optimized search
            all_rects = find_all_sum_rects_hashmap(matrix, TARGET_SUM)
            chosen = greedy_select_nonoverlap(all_rects, rows, cols, strategy='areaAsc')

            # if chosen different from last_chosen, update overlay canvas
            need_redraw = (len(chosen) != len(last_chosen)) or any(
                (chosen[i]['r1']!=last_chosen[i]['r1'] or chosen[i]['c1']!=last_chosen[i]['c1'] or
                 chosen[i]['r2']!=last_chosen[i]['r2'] or chosen[i]['c2']!=last_chosen[i]['c2']) 
                for i in range(min(len(chosen), len(last_chosen)))
            ) if last_chosen else True

            last_chosen = chosen

            if show_overlay and chosen:
                canvas = draw_overlays_canvas((region['height'], region['width']), chosen, rows, cols)
                last_canvas = canvas
                # composite overlay on top of captured preview for display (so user sees context)
                base = cv2.resize(img, (region['width'], region['height']))
                alpha = canvas[:,:,3:4].astype(float)/255.0
                fg = canvas[:,:,:3].astype(float)
                bg = base.astype(float)
                comp = (fg * alpha + bg * (1-alpha)).astype(np.uint8)
                cv2.imshow(overlay_name, comp)
            else:
                # no overlay or nothing to draw: show preview cheaply
                if last_canvas is not None and not show_overlay:
                    # if overlay hidden, show just base
                    preview = cv2.resize(img, (region['width'], region['height']))
                    cv2.imshow(overlay_name, preview)
                else:
                    preview = cv2.resize(img, (region['width'], region['height']))
                    cv2.imshow(overlay_name, preview)

            # handle key
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                if key == ord('q'):
                    running = False; break
                elif key == ord('r'):
                    print("重新选区...")
                    region = select_region_interactive()
                    if region is not None:
                        boxes = None; last_fps=None; last_matrix=None; last_chosen=[] 
                        cv2.resizeWindow(overlay_name, region['width'], region['height'])
                        cv2.moveWindow(overlay_name, region['left'], region['top'])
                elif key == ord('t'):
                    show_overlay = not show_overlay
                    print("toggle overlay ->", show_overlay)
                elif key == ord('s'):
                    single_step = True
                    print("single step triggered")

            single_step = False
            # sleep with small slices
            elapsed = time.time() - t0
            to_wait = max(0.03, CAPTURE_INTERVAL - elapsed)
            tstart = time.time()
            while time.time() - tstart < to_wait:
                time.sleep(0.02)
    finally:
        executor.shutdown(wait=False)
        cv2.destroyAllWindows()

# ------------------ small helper ------------------
def matrices_equal(a,b):
    if a is None or b is None: return False
    if len(a) != len(b): return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]): return False
        for j in range(len(a[i])):
            if int(a[i][j]) != int(b[i][j]): return False
    return True

# ------------------ run ------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted, exit.")