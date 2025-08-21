import numpy as np
import re


def parse_loop_file(filename):
    """解析回环检测结果文件"""
    loops = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if line.startswith('#') or line.startswith('BoW3D') or line.startswith('Format:') or line.startswith(
                    '=') or not line:
                continue

            # 解析数据行
            parts = line.split(', ')
            if len(parts) >= 15:  # 确保有足够的数据
                frame_id = int(parts[0])
                loop_frame_id = int(parts[1])
                detection_time = float(parts[2])

                # 解析旋转矩阵(9个值)
                rotation = [float(parts[i]) for i in range(3, 12)]

                # 解析平移向量(3个值)
                translation = [float(parts[i]) for i in range(12, 15)]

                loops.append({
                    'frame_id': frame_id,
                    'loop_frame_id': loop_frame_id,
                    'detection_time': detection_time,
                    'rotation': np.array(rotation).reshape(3, 3),
                    'translation': np.array(translation)
                })

    return loops


def create_loop_dict(loops):
    """创建以(frame_id, loop_frame_id)为键的字典"""
    loop_dict = {}
    for loop in loops:
        key = (loop['frame_id'], loop['loop_frame_id'])
        loop_dict[key] = loop
    return loop_dict


def find_matching_pairs(python_dict, cpp_dict, tolerance=2):
    """找到两个字典中匹配的回环对（带容忍度）"""
    python_keys = set(python_dict.keys())
    cpp_keys = set(cpp_dict.keys())

    # 存储匹配结果
    matched_pairs = {}  # {python_key: cpp_key}
    matched_python_keys = set()
    matched_cpp_keys = set()

    # 对每个Python回环对，寻找C++中的匹配对
    for py_frame, py_loop in python_keys:
        found_match = False

        # 在容忍范围内搜索匹配的C++回环对
        for frame_offset in range(-tolerance, tolerance + 1):
            for loop_offset in range(-tolerance, tolerance + 1):
                cpp_candidate = (py_frame + frame_offset, py_loop + loop_offset)

                if cpp_candidate in cpp_keys and cpp_candidate not in matched_cpp_keys:
                    # 找到匹配，记录映射关系
                    matched_pairs[(py_frame, py_loop)] = cpp_candidate
                    matched_python_keys.add((py_frame, py_loop))
                    matched_cpp_keys.add(cpp_candidate)
                    found_match = True
                    break

            if found_match:
                break

    # 未匹配的回环对
    python_only = python_keys - matched_python_keys
    cpp_only = cpp_keys - matched_cpp_keys

    return matched_pairs, python_only, cpp_only


def calculate_rotation_difference(R1, R2):
    """计算两个旋转矩阵的角度差（度）"""
    # 计算相对旋转矩阵
    R_diff = R1.T @ R2

    # 从旋转矩阵计算角度差
    trace = np.trace(R_diff)
    # 确保trace在有效范围内
    trace = np.clip(trace, -1, 3)
    angle = np.arccos((trace - 1) / 2)

    return np.degrees(angle)


def calculate_translation_difference(t1, t2):
    """计算两个平移向量的欧几里得距离"""
    return np.linalg.norm(t1 - t2)


def compare_loop_detections(python_file, cpp_file, tolerance=2):
    """对比两个回环检测结果文件"""
    print("开始解析文件...")

    # 解析两个文件
    python_loops = parse_loop_file(python_file)
    cpp_loops = parse_loop_file(cpp_file)

    print(f"Python检测到的回环数量: {len(python_loops)}")
    print(f"C++检测到的回环数量: {len(cpp_loops)}")

    # 创建字典便于查找
    python_dict = create_loop_dict(python_loops)
    cpp_dict = create_loop_dict(cpp_loops)

    # 找到匹配的回环对（带容忍度）
    matched_pairs, python_only, cpp_only = find_matching_pairs(python_dict, cpp_dict, tolerance)

    print(f"\n=== 回环匹配对分析（容忍度±{tolerance}） ===")
    print(f"匹配的回环对数量: {len(matched_pairs)}")
    print(f"仅Python检测到的回环对: {len(python_only)}")
    print(f"仅C++检测到的回环对: {len(cpp_only)}")

    # 显示匹配率
    total_unique = len(python_dict) + len(cpp_dict) - len(matched_pairs)
    match_rate = len(matched_pairs) / len(python_dict) * 100 if len(python_dict) > 0 else 0
    print(f"Python回环对匹配率: {match_rate:.2f}%")

    # 显示所有匹配的回环对
    if matched_pairs:
        print(f"\n所有匹配的回环对 (Python -> C++):")
        for py_key, cpp_key in sorted(matched_pairs.items()):
            if py_key == cpp_key:
                print(f"  {py_key} -> {cpp_key} (完全匹配)")
            else:
                print(f"  {py_key} -> {cpp_key} (容忍匹配)")

    # 显示所有仅Python检测到的回环对
    if python_only:
        print(f"\n所有仅Python检测到的回环对 ({len(python_only)}个):")
        for key in sorted(python_only):
            print(f"  {key}")

    # 显示所有仅C++检测到的回环对
    if cpp_only:
        print(f"\n所有仅C++检测到的回环对 ({len(cpp_only)}个):")
        for key in sorted(cpp_only):
            print(f"  {key}")

    # 对匹配的回环对进行变换和时间对比
    if matched_pairs:
        print(f"\n=== 匹配回环对的变换对比 ===")

        rotation_diffs = []
        translation_diffs = []
        time_diffs = []

        for py_key, cpp_key in matched_pairs.items():
            python_loop = python_dict[py_key]
            cpp_loop = cpp_dict[cpp_key]

            # 计算旋转差异
            rot_diff = calculate_rotation_difference(
                python_loop['rotation'],
                cpp_loop['rotation']
            )
            rotation_diffs.append(rot_diff)

            # 计算平移差异
            trans_diff = calculate_translation_difference(
                python_loop['translation'],
                cpp_loop['translation']
            )
            translation_diffs.append(trans_diff)

            # 计算时间差异
            time_diff = abs(python_loop['detection_time'] / cpp_loop['detection_time'])
            time_diffs.append(time_diff)

        rotation_diffs = np.array(rotation_diffs)
        translation_diffs = np.array(translation_diffs)
        time_diffs = np.array(time_diffs)

        print(f"旋转矩阵角度差异统计 (度):")
        print(f"  平均值: {np.mean(rotation_diffs):.4f}")
        print(f"  中位数: {np.median(rotation_diffs):.4f}")
        print(f"  最大值: {np.max(rotation_diffs):.4f}")
        print(f"  标准差: {np.std(rotation_diffs):.4f}")

        print(f"\n平移向量差异统计 (单位长度):")
        print(f"  平均值: {np.mean(translation_diffs):.4f}")
        print(f"  中位数: {np.median(translation_diffs):.4f}")
        print(f"  最大值: {np.max(translation_diffs):.4f}")
        print(f"  标准差: {np.std(translation_diffs):.4f}")

        print(f"\n检测时间差异统计 (倍):")
        print(f"  平均值: {np.mean(time_diffs):.4f}")
        print(f"  中位数: {np.median(time_diffs):.4f}")
        print(f"  最大值: {np.max(time_diffs):.4f}")
        print(f"  标准差: {np.std(time_diffs):.4f}")

        # 显示差异最大的几个案例
        print(f"\n=== 差异最大的回环对 ===")

        # 旋转差异最大的5个
        max_rot_indices = np.argsort(rotation_diffs)[-5:]
        print("旋转差异最大的5个回环对:")
        for idx in reversed(max_rot_indices):
            py_key, cpp_key = list(matched_pairs.items())[idx]
            print(f"  {py_key} -> {cpp_key}: {rotation_diffs[idx]:.4f}度")

        # 平移差异最大的5个
        max_trans_indices = np.argsort(translation_diffs)[-5:]
        print("\n平移差异最大的5个回环对:")
        for idx in reversed(max_trans_indices):
            py_key, cpp_key = list(matched_pairs.items())[idx]
            print(f"  {py_key} -> {cpp_key}: {translation_diffs[idx]:.4f}")

        # 时间差异最大的5个
        max_time_indices = np.argsort(time_diffs)[-5:]
        print("\n时间差异最大的5个回环对:")
        for idx in reversed(max_time_indices):
            py_key, cpp_key = list(matched_pairs.items())[idx]
            print(f"  {py_key} -> {cpp_key}: {time_diffs[idx]:.4f}倍")


# 主函数
if __name__ == "__main__":
    python_file = "/home/wzj/pan1/bow3d-python/result/loop-python.txt"
    cpp_file = "/home/wzj/pan1/BoW3D_ws/result/loop-c++.txt"
    try:
        compare_loop_detections(python_file, cpp_file)
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")