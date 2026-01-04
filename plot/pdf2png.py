from pdf2image import convert_from_path
import os

def pdf_to_png_simple(pdf_path, output_dir="pdf2png_output", dpi=200):
    """
    简易PDF转PNG（批量转换所有页面）
    :param pdf_path: PDF文件的路径（绝对路径/相对路径）
    :param output_dir: 输出PNG图片的文件夹
    :param dpi: 图片分辨率（dpi越高，图片越清晰，文件越大）
    :return: 无返回值，图片保存到输出文件夹
    """
    # 1. 创建输出文件夹（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出文件夹：{output_dir}")

    try:
        # 2. 转换PDF为图片对象列表（每页对应一个图片对象）
        print(f"正在读取PDF文件：{pdf_path}")
        images = convert_from_path(
            pdf_path=pdf_path,
            dpi=dpi,  # 分辨率设置
            fmt="png",  # 输出格式指定为PNG
            output_folder=None,  # 先不直接保存，后续自定义命名
            thread_count=2  # 多线程加速转换
        )

        # 3. 遍历图片对象，保存为PNG文件
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]  # 获取PDF文件名（不含后缀）
        for idx, img in enumerate(images, start=1):
            # 自定义图片命名：PDF名_页码.png
            png_filename = f"{pdf_name}_page{idx}.png"
            png_path = os.path.join(output_dir, png_filename)
            img.save(png_path, "PNG")
            print(f"已保存：{png_path}")

        print(f"\n转换完成！共转换 {len(images)} 页，保存至：{os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"转换失败！错误信息：{e}")

# 调用示例
if __name__ == "__main__":
    # 替换为你的PDF文件路径
    target_pdf = r"D:\python_resp\MOE-AHL_naster\plot\主图.pdf"
    # 调用转换函数
    pdf_to_png_simple(pdf_path=target_pdf, dpi=300)