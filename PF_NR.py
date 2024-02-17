import numpy as np
import scipy.io as sio
import pandas as pd


class PowerFlow():
    def __init__(self, branch, gen, bus) -> None:

        self.baseMVA = 20 #MVA

        # self.bus = dict(zip(BusKey,mpc['bus'][0,0].T))
        self.bus = bus
        self.gen = gen
        self.branch = branch
        self.bus_i = self.bus['bus_i'] #kW	KVAR


        self.nodeNum = len(self.bus['bus_i'])
        self.branchNum = len(self.branch['fbus'])

        pass

    # 由广义节点-支路关联矩阵和广义支路导纳矩阵生成节点导纳矩阵
    # Ybus = N_B_Aij_Generalized @ Ybranch_Generalized @ N_B_Aij_Generalized.T
    def _Get_Ybus(self):
        self._GetN_B_Aij_Generalized()
        self._Get_Ybranch_Generalized()
        self.Ybus = self.N_B_Aij_Generalized @ self.Ybranch_Generalized @ self.N_B_Aij_Generalized.T
        Ysh = (self.bus['Gs'] + 1j * self.bus['Bs'])/self.baseMVA
        self.Ybus = self.Ybus + np.diag(Ysh)


    # 生成广义 节点-支路 关联矩阵 （考虑参考节点）shape = (nodeNum,branchNum+nodeNum)
    def _GetN_B_Aij_Generalized(self):    # 广义的节点支路关联矩阵、考虑变压器变比
        self.N_B_Aij_Generalized = np.zeros((self.nodeNum,self.branchNum+self.nodeNum))
        ratio = self.branch['ratio']
        ratio = np.where(ratio == 0,1,ratio)
        for i in range(self.branchNum):
            self.N_B_Aij_Generalized[int(self.branch["fbus"][i]),i] = 1/ratio[i]
            self.N_B_Aij_Generalized[int(self.branch["tbus"][i]),i] = -1
        for i in range(self.nodeNum):
            self.N_B_Aij_Generalized[i,self.branchNum+i] = 1

    # 生成广义 支路导纳矩阵 （考虑参考节点） shape = (branchNum+nodeNum,branchNum+nodeNum)
    def _Get_Ybranch_Generalized(self):  # 广义的支路导纳矩阵、考虑变压器变比
        self.Ybranch_Generalized= np.zeros((self.branchNum+self.nodeNum,self.branchNum+self.nodeNum),dtype=complex)
        self.Ybranch_Generalized[:self.branchNum,:self.branchNum] = np.diag(1/(self.branch['r'] + 1j*self.branch['x']))*self.branch['status']
        for i in range(self.nodeNum):
            a = np.where(self.branch['fbus'] == i,1,0)+np.where(self.branch['tbus'] == i,1,0)   # 生成一个mask 用于计算对地支路的电导
            self.Ybranch_Generalized[self.branchNum+i,self.branchNum+i] = np.sum(a*(1j * self.branch['b'] / 2))

    # 生成节点 节点-之路 关联矩阵 （不考虑参考节点）shape = (nodeNum,branchNum)
    def _GetN_B_Aij(self):
        self.N_B_Aij = np.zeros((self.nodeNum,self.branchNum))
        for i in range(self.branchNum):
            self.N_B_Aij[int(self.branch["fbus"][i]),i] = 1
            self.N_B_Aij[int(self.branch["tbus"][i]),i] = -1

    # result后处理
    def _ProcessRes(self,S_Gen,S_load,typekey):
        S_Gen = S_Gen*self.baseMVA
        S_load = S_load*self.baseMVA
        Ibus, Sbus = self._make_Sbus(self.V_nr)
        Sbus = Sbus*self.baseMVA
        S_Gen = Sbus+S_load
        self.result_bus = np.vstack([self.bus_i,
                                     np.abs(self.V_nr).real,
                                     np.angle(self.V_nr,deg=True).real,
                                     np.abs(Ibus).real,
                                     np.angle(Ibus,deg=True).real,
                                     S_Gen.real,
                                     S_Gen.imag,
                                     S_load.real,
                                     S_load.imag]).T
        self.result_bus_col = ['bus_i','Vabs','Vangle','Iabs','Iangle','Gen_P','Gen_Q','Pload','Qload']
        self.result_bus = np.round(self.result_bus,4)
        #self._make_cost(self.result_bus[:,5],typekey)

        # 计算网损：总负荷功率与总发电功率之差的负值
        losses = np.sum(S_load.real) - np.sum(S_Gen.real)

        return losses


    # 从节点电压向量  ->  eifi（去除V_delta节点）  V = ei + jfi
    def _V2eifi(self,V,typekey):
        V = np.delete(V,typekey[0],None)
        ei = np.real(V)
        fi = np.imag(V)
        return np.concatenate((ei,fi))
    
    # 从eifi还原为节点电压 （包括V_delta节点）V = ei + jfi
    def _eifi2V(self,eifi,typekey):
        V = eifi[:self.nodeNum-1] + 1j*eifi[self.nodeNum-1:]
        V = np.insert(V,typekey[0],self.V0[typekey[0][0]])
        return V     

    # 识别节点类型 并排序
    def _bus_types(self):

        types_ref = np.where(self.bus['type'] == 3)[0]
        types_pv = np.where(self.bus['type'] == 2)[0]
        types_pq = np.where(self.bus['type'] == 1)[0]
        types_ref = np.sort(types_ref)
        types_pv = np.sort(types_pv)
        types_pq = np.sort(types_pq)
        return [types_ref,types_pv,types_pq]

    # 计算节点注入功率  S_in - V @  Y.conj @ V.conj = = 0!
    def _make_Sbus(self,V):
        Ibus = self.Ybus @ V
        Sbus = np.diag(V).conj() @ self.Ybus @ V
        Sbus = Sbus.conj()
        return Ibus,Sbus
    
    # 计算节点外部注入功率   =（发电机有功无功注入 - 负荷有功无功注入）/基准电压 注意单位 原数据是kW 和KVA
    def _make_Sin(self):#
        # 转换负载功率为pu
        S_load = (self.bus['Pd'] + 1j * self.bus['Qd']) / 1000 / self.baseMVA

        # 转换发电机功率为pu
        gen_on = np.where(self.gen['status'] == 1)[0]
        Gen = (self.gen['Pg'][gen_on] + 1j * self.gen['Qg'][gen_on]) / 1000 / self.baseMVA
        S_Gen = np.zeros(self.nodeNum, dtype=complex)
        for i, seq in enumerate(self.gen['gen_bus'][gen_on]):
            S_Gen[seq] += Gen[i]

        self.Sin = S_Gen - S_load

        return S_Gen,S_load

    def _make_PspQsp(self,typekey):

        Key = np.concatenate((typekey[1],typekey[2]))
        Key=np.sort(Key)
        Psp = self.Sin.real[Key]
        Qsp = self.Sin.imag[typekey[2]]
        return Psp,Qsp
    
    def _make_fx(self,V,typekey):   # f(x)   :  P,Q,V 

        # 计算F(x)的值（直接通过网络矩阵计算速度快）
        # S_in - V @  Y.conj @ V.conj = = 0!
        Ibus ,Sbus = self._make_Sbus(V)
        Sbus = self.Sin - Sbus
        # Psp,Qsp = self._make_PspQsp(typekey)
        self.Sbustext = Sbus
        ai = np.real(Ibus) ## 难道问题出在了这里？
        bi = np.imag(Ibus)

        ei = np.real(V)
        fi = np.imag(V)

        Key = np.concatenate((typekey[1],typekey[2]))
        Key = np.sort(Key)

        P_delta = np.delete(np.real(Sbus),typekey[0],None)
        Q_delta = np.imag(Sbus)[typekey[2]]
        VV_delta = self.bus['Vm'][typekey[1]]**2-ei[typekey[1]]**2-fi[typekey[1]]**2

        ai = np.expand_dims(ai, axis=-1)  # 转换为列向量 
        bi = np.expand_dims(bi, axis=-1)
        ei = np.expand_dims(ei, axis=-1)
        fi = np.expand_dims(fi, axis=-1)
        return np.concatenate([P_delta,Q_delta,VV_delta],axis=0),ai,bi,ei,fi
    
    
    def _make_Jac(self,ai,bi,ei,fi,G,B,typekey):
        # 这里计算雅克比矩阵，直角坐标形式。
        # 先计算所有节点的雅克比矩阵，然后再删除不需要的节点行列。具体参考高等电网络分析第三版 P182 牛拉法潮流计算章节
        Jac = np.zeros(((self.nodeNum-1)*2,(self.nodeNum-1)*2),dtype=complex)

        Hii = np.diag(-ai-(np.diag(G)*ei+np.diag(B)*fi))
        H = -(G*ei+B*fi)
        np.fill_diagonal(H,Hii)

        Nii = np.diag(-bi-(np.diag(G)*fi-np.diag(B)*ei))
        N = -(G*fi-B*ei)
        np.fill_diagonal(N,Nii)

        Key = np.concatenate((typekey[1],typekey[2]))
        Key=np.sort(Key)

        H = np.delete(H,typekey[0],axis=1)
        N = np.delete(N,typekey[0],axis=1)
        Jac[0:self.nodeNum-1,:] = np.concatenate((H,N),axis=1)[Key,:]

        Mii = np.diag(bi+(np.diag(B)*ei-np.diag(G)*fi))
        M = -(G*fi-B*ei)
        np.fill_diagonal(M,Mii)

        Lii = np.diag(-ai+(np.diag(B)*fi+np.diag(G)*ei))
        L = (G*ei+B*fi)
        np.fill_diagonal(L,Lii)

        Key = typekey[2]
        Key=np.sort(Key)

        M = np.delete(M,typekey[0],axis=1)
        L = np.delete(L,typekey[0],axis=1)

        Jac[self.nodeNum-1:self.nodeNum-1+typekey[2].shape[0],:] = np.concatenate((M,L),axis=1)[Key,:]

        R = np.diag(-2*ei.T[0])

        S = np.diag(-2*fi.T[0])

        Key = typekey[1]
        Key=np.sort(Key)

        R = np.delete(R,typekey[0],axis=1)
        S = np.delete(S,typekey[0],axis=1)
        Jac[self.nodeNum-1+typekey[2].shape[0]:,:] = np.concatenate((R,S),axis=1)[Key,:]
        return Jac


    # 检查无功是否超出 注意单位
    def _check_and_adjust_qg(self):
        for i, gen_bus in enumerate(self.gen['gen_bus']):
            # 计算无功功率
            V_gen = self.V_nr[gen_bus]  # 发电机节点的电压
            S_gen = V_gen * np.conj(self.Ybus[gen_bus, :] @ self.V_nr)  # 发电机节点的复功率
            Q_gen = S_gen.imag * self.baseMVA  # 无功功率，转换为物理单位

            # self.gen['Qmax'] 和 self.gen['Qmin'] 是以 kVAR 为单位
            Qmax_MVAR = self.gen['Qmax'][i] / 1000  # 转换为 MVAR
            Qmin_MVAR = self.gen['Qmin'][i] / 1000  # 转换为 MVAR

            if Q_gen > Qmax_MVAR:
                Q_gen = Qmax_MVAR
            elif Q_gen < Qmin_MVAR:
                Q_gen = Qmin_MVAR

            self.gen['Qg'][i] = Q_gen * 1000  # 将结果转换回 kVAR 存储

    # 运行牛拉法潮流
    def RunPF(self, el=1e-6, lmax=100):
        self._Get_Ybus()
        typekey = self._bus_types()  # [ref, pv, pq]

        # 初始化电压
        self.V0 = self.bus['Vm'] * np.exp(1j * np.pi / 180 * self.bus['Va'])
        self.V_nr = self.V0

        # 初始化注入功率
        S_Gen, S_load = self._make_Sin()

        # 迭代开始
        lnum = 0
        while True:
            lnum += 1  # 迭代次数加1

            # 计算功率不平衡和雅可比矩阵
            Fx, ai, bi, ei, fi = self._make_fx(self.V_nr, typekey)
            Jac = self._make_Jac(ai, bi, ei, fi, self.Ybus.real, self.Ybus.imag, typekey)

            # 求解增量
            delta = np.linalg.solve(Jac, -Fx)
            self.eifi = self._V2eifi(self.V_nr, typekey) + delta
            self.V_nr = self._eifi2V(self.eifi, typekey)

            # 检查并调整PV节点的无功功率
            self._check_and_adjust_qg()

            # 检查收敛条件
            if np.max(np.abs(Fx)) < el:
                losses = self._ProcessRes(S_Gen, S_load, typekey)  # 计算网损
                return losses, self.V_nr  # 返回网损

            if lnum > lmax:
                print("[warn]: Please Mind! The N-R did not converge!!")
                break  # 未能收敛，跳出循环

        return None, self.V_nr


def Res2Excel_bus(PF, filename, sheetname):
    writer = pd.ExcelWriter(filename)
    df = pd.DataFrame(PF.result_bus, columns=PF.result_bus_col)
    df.to_excel(writer, sheet_name=sheetname)
    writer.save()
    writer.close()

def Res2Excel_branch(PF, filename, sheetname):
    writer = pd.ExcelWriter(filename)
    df = pd.DataFrame(PF.result_branch, columns=PF.result_branch_col)
    df.to_excel(writer, sheet_name=sheetname)

    writer.save()
    writer.close()

