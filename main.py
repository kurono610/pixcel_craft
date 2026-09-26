import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

def apply_color_quantization(img, k=16):
    """画像の色数を指定した色数(k色)に制限して減色する（中間色をなくす）"""
    # アルファチャンネル（透過）がある場合はRGB部分のみ減色
    has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False
    
    if has_alpha:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img

    # k-meansクラスタリング用にデータを変換
    data = np.float32(bgr).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    
    # 色のクラスタリング
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 代表色に置き換え
    centers = np.uint8(centers)
    quantized_bgr = centers[labels.flatten()].reshape(bgr.shape)

    if has_alpha:
        # アルファチャンネルを結合し直す
        return cv2.merge([quantized_bgr, alpha])
    return quantized_bgr

def process_image(input_path, output_path, num_dots=64, num_colors=16):
    """アルファ透過・くっきり補間・減色に対応した画像処理"""
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, "画像の読み込みに失敗しました。"

    height, width = img.shape[:2]

    # アスペクト比を維持して縦のドット数を計算
    aspect_ratio = height / width
    small_w = max(1, num_dots)
    small_h = max(1, int(num_dots * aspect_ratio))

    # 【重要】縮小時も INTER_NEAREST を使うことでぼやけを完全に防止
    small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    # 減色処理で色のグラデーションをパキッとさせる（必要に応じて）
    if num_colors > 0:
        small_img = apply_color_quantization(small_img, k=num_colors)

    # 元サイズに拡大
    pixel_art = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(output_path, pixel_art)
    return True, f"保存完了: {output_path}"

def process_video(input_path, output_path, num_dots=64, num_colors=16):
    """動画のドット絵化処理"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False, "動画の読み込みに失敗しました。"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    aspect_ratio = height / width
    small_w = max(1, num_dots)
    small_h = max(1, int(num_dots * aspect_ratio))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 縮小・拡大ともに INTER_NEAREST
        small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        
        # 減色処理（動画の場合は計算コスト削減のためオプション）
        if num_colors > 0:
            small_frame = apply_color_quantization(small_frame, k=num_colors)

        pixel_frame = cv2.resize(small_frame, (width, height), interpolation=cv2.INTER_NEAREST)
        out.write(pixel_frame)

    cap.release()
    out.release()
    return True, f"保存完了: {output_path}"

def get_settings_from_dialog():
    """ドット数と色数を設定する"""
    dialog_root = tk.Tk()
    dialog_root.withdraw()
    dialog_root.attributes('-topmost', True)

    num_dots = simpledialog.askinteger(
        "ドット数の選択",
        "画面横方向に並べるドット数を指定してください (例: 32〜128):",
        initialvalue=64, minvalue=8, maxvalue=1000
    )
    if not num_dots:
        dialog_root.destroy()
        return None, None

    num_colors = simpledialog.askinteger(
        "使用色数の制限（減色）",
        "画面全体で使用する最大色数を指定してください:\n"
        "（※中間色が消えてパキッとした質感になります）\n\n"
        "・ 8 〜 16  : 昔のレトロゲーム風\n"
        "・ 32 〜 64 : 鮮やかなドット絵\n"
        "・ 0       : 減色なし（元の色数を維持）",
        initialvalue=16, minvalue=0, maxvalue=256
    )
    
    dialog_root.destroy()
    return num_dots, num_colors

def main():
    root = tk.Tk()
    root.withdraw()

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

    file_path = filedialog.askopenfilename(
        title="処理したい画像または動画を選択してください",
        filetypes=[
            ("画像ファイル", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.wmv")
        ]
    )

    if not file_path:
        return

    num_dots, num_colors = get_settings_from_dialog()
    if num_dots is None:
        return

    ext = os.path.splitext(file_path)[1].lower()
    dir_name, file_name = os.path.split(file_path)
    base_name, _ = os.path.splitext(file_name)

    suffix = f"_{num_dots}dots"
    if num_colors and num_colors > 0:
        suffix += f"_{num_colors}colors"

    if ext in image_exts:
        output_path = os.path.join(dir_name, f"{base_name}{suffix}{ext}")
        print("画像ファイルを処理中...")
        success, message = process_image(file_path, output_path, num_dots, num_colors)

    elif ext in video_exts:
        output_path = os.path.join(dir_name, f"{base_name}{suffix}.mp4")
        print("動画ファイルを処理中...")
        success, message = process_video(file_path, output_path, num_dots, num_colors)

    else:
        messagebox.showerror("エラー", "サポートされていないファイル形式です。")
        return

    if success:
        messagebox.showinfo("完了", message)
    else:
        messagebox.showerror("エラー", message)

if __name__ == "__main__":
    main()