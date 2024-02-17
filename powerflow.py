from gridinfo import branch, bus, gen, mapped_nodedata_dict, node_mapping, mapped_pv_dict, mapped_wt_dict
import numpy as np
from PF_NR import PowerFlow

def powerflow(EVload, Pnet_mic, Pnet, Psg):

    # 初始化存储一天中每小时电压偏差均方根值的列表
    daily_vdrms_values = []
    # 初始化总损耗为0
    total_losses = 0

    for i in range(48):  # 修正循环，从0到47
        # 获取当前小时的电动汽车负荷
        Pev_i = EVload[i]
        # 更新bus字典中的Pd和Qd
        # 假设nodedata_dict[i]是一个二维数组或类似结构，其中包含了当前小时所有节点的Pd和Qd值
        bus["Pd"] = mapped_nodedata_dict[i][:, 1]  # 更新Pd为nodedata_dict的第2列
        bus["Qd"] = mapped_nodedata_dict[i][:, 2]  # 更新Qd为nodedata_dict的第3列
        bus["Pd"] += Pev_i

        # 获取特定时间步长的发电机出力
        Psg_i = Psg[i]
        # 定义目标 gen_bus 编号
        target_gen_bus = [101, 201, 301, 401]
        # 使用node_mapping字典将节点编号映射为索引
        mapped_target_gen_bus = [node_mapping[bus1] for bus1 in target_gen_bus]
        # 遍历目标 gen_bus 值
        for idx, target_bus in enumerate(mapped_target_gen_bus):
            # 找到对应 gen_bus 的索引
            index = np.where(gen['gen_bus'] == target_bus)[0]
            if len(index) > 0:  # 确保找到了匹配的 gen_bus
                # 更新 Pg 值，假设Psg_i[idx]是对应于target_bus的发电输出
                gen['Pg'][index] = Psg_i[idx]
        # 更新pv
        for row in mapped_pv_dict[i]:
            pv_bus, pvi = row  # 第一列是 gen_bus，第二列是 capacity
            # 找到对应 gen_bus 的索引
            index = np.where(gen['gen_bus'] == pv_bus)[0]
            if len(index) > 0:  # 确保找到了匹配的 gen_bus
                # 更新 Pg 值
                gen['Pg'][index] = pvi
        # 更新wt
        for row in mapped_wt_dict[i]:
            wt_bus, wti = row  # 第一列是 gen_bus，第二列是 capacity
            # 找到对应 gen_bus 的索引
            index = np.where(gen['gen_bus'] == wt_bus)[0]
            if len(index) > 0:  # 确保找到了匹配的 gen_bus
                # 更新 Pg 值
                gen['Pg'][index] = wti

        # 初始化status数组，默认所有连接都是激活的，即状态为1
        branch['status'] = np.ones_like(branch['fbus'])

        # 更新微电网间功率，使用节点到索引的映射转换节点对
        # 定义节点对应的微电网关系
        microgrid_relations = [
            (102, 201, 1, 2),
            (104, 301, 1, 3),
            (208, 401, 2, 4),
            (205, 310, 2, 3),
            (318, 404, 3, 4)
        ]
        for source, target, mg_source, mg_target in microgrid_relations:
            # 使用node_mapping来获取源节点和目标节点的索引
            source_index = node_mapping[source]
            target_index = node_mapping[target]
            # 从Pnet_mic获取对应微电网间的功率流
            power_flow = Pnet_mic[i].get((mg_source, mg_target), 0)

            # 如果power_flow大于0，更新gen和bus字典
            if power_flow > 0: # mg_source买电 流入mg_source电网  mg_source发电机 target负荷
                # 更新gen字典中对应的发电机出力
                gen_indices = [idx for idx, bus_id in enumerate(gen['gen_bus']) if bus_id == source_index]
                for idx in gen_indices:
                    gen['Pg'][idx] += power_flow

                # 更新bus字典中对应的负载
                bus_indices = [idx for idx, bus_id in enumerate(bus['bus_i']) if bus_id == target_index]
                for idx in bus_indices:
                    bus['Pd'][idx] += power_flow
            else: # mg_source卖电 流出mg_source电网  mg_source负荷 target发电机
                # 更新gen字典中对应的发电机出力
                gen_indices = [idx for idx, bus_id in enumerate(gen['gen_bus']) if bus_id == target_index]
                for idx in gen_indices:
                    gen['Pg'][idx] += power_flow

                # 更新bus字典中对应的负载
                bus_indices = [idx for idx, bus_id in enumerate(bus['bus_i']) if bus_id == source_index]
                for idx in bus_indices:
                    bus['Pd'][idx] += power_flow

        # 更新主电网功率，使用节点到索引的映射转换节点对
        # 定义节点对应的微电网关系
        grid_relations = [
            (101, 1),
            (201, 2),
            (301, 3),
        ]
        for node, mg_target in microgrid_relations:
            # 使用node_mapping来获取源节点和目标节点的索引
            node_index = node_mapping[node]
            # 从Pnet获取对应微电网间的功率流
            power = Pnet[i].get(mg_target, 0)

            # 如果power_flow大于0，更新gen和bus字典
            if power > 0: # 从主电网买电 虚拟发电机
                # 更新gen字典中对应的发电机出力
                gen_indices = [idx for idx, bus_id in enumerate(gen['gen_bus']) if bus_id == node_index]
                for idx in gen_indices:
                    gen['Pg'][idx] += power
            else: # 向主电网卖电 虚拟负载
                # 更新bus字典中对应的负载
                bus_indices = [idx for idx, bus_id in enumerate(bus['bus_i']) if bus_id == node_index]
                for idx in bus_indices:
                    bus['Pd'][idx] += power

        a = PowerFlow(branch, gen, bus)
        losses, voltage_pu = a.RunPF()
        # 累加当前的损耗
        total_losses += losses
        # # 期望的电压标幺值
        # expected_voltage_pu = 1.0
        # # 计算电压偏差
        # voltage_deviation = voltage_pu - expected_voltage_pu
        # # 计算每小时电压偏差的均方根值并存储
        # Vdrms = np.sqrt(np.mean(voltage_deviation ** 2))
        # daily_vdrms_values.append(Vdrms)
    print("潮流计算结束")
    return total_losses

    # # 将结果转换为NumPy数组以便进行进一步的分析
    # daily_vdrms_values = np.array(daily_vdrms_values)
    #
    # # 设定阈值，计算超过阈值的时段比例
    # threshold = 0.05 #5%偏差
    # over_threshold_ratio = np.sum(daily_vdrms_values > threshold) / len(daily_vdrms_values)
    #
    # return over_threshold_ratio


    # # 遍历branch中的每条连接
        # for idx in range(len(branch['fbus'])):
        #     fbus = branch['fbus'][idx]
        #     tbus = branch['tbus'][idx]




        #
        # # 初始化Pnet_flows为空字典
        # Pnet_flows = {}
        #
        # # 对于当前时间步i，基于映射关系填充Pnet_flows
        # for source, target, mg_source, mg_target in microgrid_relations:
        #     # 使用node_mapping来获取源节点和目标节点的索引
        #     source_index = node_mapping[source]
        #     target_index = node_mapping[target]
        #
        #     # 从Pnet_mic获取对应微电网间的功率流，如果不存在，则默认为0
        #     power_flow = Pnet_mic[i].get((mg_source, mg_target), 0)
        #
        #     # 填充Pnet_flows，键是源节点和目标节点的索引对
        #     Pnet_flows[(source_index, target_index)] = power_flow





