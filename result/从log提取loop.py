import re
import numpy as np


def extract_loop_detection_results(log_file_path, output_file_path):
    """
    从BoW3D运行日志中提取回环检测结果

    Args:
        log_file_path: 输入日志文件路径
        output_file_path: 输出结果文件路径
    """

    loop_results = []

    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按帧分割内容
    frame_sections = re.split(r'----+', content)

    for section in frame_sections:
        if '检测到回环帧' in section:
            try:
                # 提取帧ID
                frame_match = re.search(r'处理帧 (\d+):', section)
                if not frame_match:
                    continue
                frame_id = int(frame_match.group(1))

                # 提取回环帧ID
                loop_frame_match = re.search(r'检测到回环帧 (\d+)', section)
                if not loop_frame_match:
                    continue
                loop_frame_id = int(loop_frame_match.group(1))

                # 提取检测时间
                detection_time_match = re.search(r'检测时间: ([\d.]+)s', section)
                if not detection_time_match:
                    continue
                detection_time = float(detection_time_match.group(1))

                # 提取旋转矩阵
                rotation_pattern = r'回环相对旋转矩阵:\s*\[([-\d.\s]+)\]\s*\[([-\d.\s]+)\]\s*\[([-\d.\s]+)\]'
                rotation_match = re.search(rotation_pattern, section)
                if not rotation_match:
                    continue

                # 解析旋转矩阵的三行
                row1 = [float(x) for x in rotation_match.group(1).split()]
                row2 = [float(x) for x in rotation_match.group(2).split()]
                row3 = [float(x) for x in rotation_match.group(3).split()]

                rotation_matrix = row1 + row2 + row3  # 展平为9个值

                # 提取平移向量
                translation_pattern = r'回环相对平移向量:\s*\[([-\d.\s]+)\]'
                translation_match = re.search(translation_pattern, section)
                if not translation_match:
                    continue

                translation = [float(x) for x in translation_match.group(1).split()]

                # 保存结果
                result = {
                    'frame_id': frame_id,
                    'loop_frame_id': loop_frame_id,
                    'detection_time': detection_time,
                    'rotation_matrix': rotation_matrix,
                    'translation': translation
                }
                loop_results.append(result)

            except Exception as e:
                print(f"处理帧时出错: {e}")
                continue

    # 写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # 写入文件头
        f.write("BoW3D Loop Detection Results - Sequence 00\n")
        f.write("Format: FrameID, LoopFrameID, DetectionTime(s), Rotation_Matrix(9_values), Translation(3_values)\n")
        f.write("=========================================\n")

        # 写入每个回环检测结果
        for result in loop_results:
            # 格式化旋转矩阵（9个值）
            rotation_str = ', '.join([f"{val:.5f}" for val in result['rotation_matrix']])
            # 格式化平移向量（3个值）
            translation_str = ', '.join([f"{val:.5f}" for val in result['translation']])

            # 写入一行数据
            line = f"{result['frame_id']}, {result['loop_frame_id']}, {result['detection_time']:.4f}, {rotation_str}, {translation_str}\n"
            f.write(line)

    print(f"提取完成！")
    print(f"总共提取到 {len(loop_results)} 个回环检测结果")
    print(f"结果已保存到: {output_file_path}")

    return loop_results


def print_statistics(loop_results):
    """打印统计信息"""
    if not loop_results:
        print("没有找到回环检测结果")
        return

    print("\n=== 统计信息 ===")
    print(f"回环检测总数: {len(loop_results)}")

    # 检测时间统计
    detection_times = [r['detection_time'] for r in loop_results]
    print(f"平均检测时间: {np.mean(detection_times):.4f}s")
    print(f"最小检测时间: {np.min(detection_times):.4f}s")
    print(f"最大检测时间: {np.max(detection_times):.4f}s")

    # 平移距离统计
    translation_distances = []
    for r in loop_results:
        trans = r['translation']
        distance = np.sqrt(trans[0] ** 2 + trans[1] ** 2 + trans[2] ** 2)
        translation_distances.append(distance)

    print(f"平均平移距离: {np.mean(translation_distances):.3f}m")
    print(f"最小平移距离: {np.min(translation_distances):.3f}m")
    print(f"最大平移距离: {np.max(translation_distances):.3f}m")

    # 帧间距统计
    frame_gaps = [r['frame_id'] - r['loop_frame_id'] for r in loop_results]
    print(f"平均帧间距: {np.mean(frame_gaps):.1f}")
    print(f"最小帧间距: {np.min(frame_gaps)}")
    print(f"最大帧间距: {np.max(frame_gaps)}")


if __name__ == "__main__":
    # 使用示例
    log_file_path = "/home/wzj/pan1/bow3d-python/result/log.txt"
    output_file_path = "/result/loop-python.txt"

    # 提取回环检测结果
    loop_results = extract_loop_detection_results(log_file_path, output_file_path)

    # 打印统计信息
    print_statistics(loop_results)

    # 显示前5个结果作为示例
    if loop_results:
        print("\n=== 前5个回环检测结果示例 ===")
        for i, result in enumerate(loop_results[:5]):
            print(f"结果 {i + 1}:")
            print(f"  当前帧: {result['frame_id']}, 回环帧: {result['loop_frame_id']}")
            print(f"  检测时间: {result['detection_time']:.4f}s")
            print(f"  旋转矩阵: {result['rotation_matrix']}")
            print(f"  平移向量: {result['translation']}")
            trans_distance = np.sqrt(sum(x ** 2 for x in result['translation']))
            print(f"  平移距离: {trans_distance:.3f}m")
            print()