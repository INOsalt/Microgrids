import numpy as np
import scipy.io as sio
import pandas as pd

class PowerFlow():
    def __init__(self,dir) -> None:
        mpc = sio.loadmat(dir)[dir.split('.')[0]]
        
        BusKey = ["bus_i", "type", "Pd", "Qd", "Gs", "Bs", "area", "Vm", "Va", "baseKV", "zone", "Vmax", "Vmin"]
        GenKey = ["gen_bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", "Pmax", "Pmin", "Pc1", "Pc2", "Qc1min", "Qc1max", "Qc2min", "Qc2max", "ramp_agc", "ramp_10", "ramp_30", "ramp_q", "apf"]
        BranchKey = ["fbus", "tbus", "r", "x", "b", "rateA", "rateB", "rateC", "ratio", "angle", "status", "angmin", "angmax"]
        # ["2", "startup", "shutdown", "n", "x1", "y1", "..."]

        self.version = mpc['version'][0,0]
        self.baseMVA =mpc['baseMVA'][0,0][0,0]
        self.gencost = mpc['gencost'][0,0]

        # try
        # self.bus_name = mpc['bus_name'][0,0]

        self.bus = dict(zip(BusKey,mpc['bus'][0,0].T))
        self.gen = dict(zip(GenKey,mpc['gen'][0,0].T))
        self.branch = dict(zip(BranchKey,mpc['branch'][0,0].T))

        self.bus_i = self.bus['bus_i']
        # 替换节点编号到0开始的编号
        bus_i_dict = dict(zip(self.bus_i,range(len(self.bus_i))))

        self.branch['fbus'] = np.array([bus_i_dict[i] for i in self.branch['fbus']])
        self.branch['tbus'] = np.array([bus_i_dict[i] for i in self.branch['tbus']])
        self.gen['gen_bus'] = np.array([bus_i_dict[i] for i in self.gen['gen_bus']])
        self.bus['bus_i'] = np.array([bus_i_dict[i] for i in self.bus['bus_i']])

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

    def _make_cost(self,P_gen,typekey):
        # 生成发电机成本函数
        self.cost = 0
        Key = np.concatenate((typekey[0],typekey[1])) # 生成发电机类型的key
        Key = np.sort(Key)
        P_gen = P_gen[Key]
        self.Cost = np.sum(self.gencost[:,4] * P_gen**2+self.gencost[:,5] * P_gen+self.gencost[:,6])

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
        self._make_cost(self.result_bus[:,5],typekey)
        
        # self.result_branch_col = ['From','To','From P','From Q','To P','To Q','Loss P','Loss Q']
        # self._GetN_B_Aij()

        # # Ybranch = self.N_B_Aij_Generalized.T @ self.Ybus @ self.N_B_Aij
        # V_branch = self.N_B_Aij_Generalized.T @ self.V_nr
        # I_branch = self.Ybranch_Generalized @ V_branch
        # S_branch = V_branch * np.conj(I_branch) * 100
        # # self.result_branch = np.vstack([self.branch['fbus'],
        # #                                 self.branch['tbus'],
        # #                                 S_branch.real,
        # #                                 S_branch.imag,
        # #                                 S_branch.real,
        # #                                 S_branch.imag,
        # #                                 S_branch.real,
        # #                                 S_branch.imag]).T
        pass

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
    
    # 计算节点外部注入功率   =（发电机有功无功注入 - 负荷有功无功注入）/基准电压 （与Matpower 格式的数据换算有关）
    def _make_Sin(self):
        S_load = (self.bus['Pd'] + 1j * self.bus['Qd'])

        gen_on = np.where(self.gen['status'] == 1)[0]
        Gen = self.gen['Pg'][gen_on] + 1j * self.gen['Qg'][gen_on]
        S_Gen = np.zeros(self.nodeNum, dtype=complex)
        for i,seq in enumerate(self.gen['gen_bus'][gen_on]) :
            S_Gen[seq] = S_Gen[seq] + Gen[i]

        S_load=S_load/self.baseMVA
        S_Gen=S_Gen/self.baseMVA
        self.Sin = S_Gen-S_load
        return S_Gen,S_load

    # 效率低，不用
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
        Sbus = self.Sin -  Sbus
        # Psp,Qsp = self._make_PspQsp(typekey)
        self.Sbustext = Sbus
        ai = np.real(Ibus) ## 难道问题出在了这里？
        bi = np.imag(Ibus)

        ei = np.real(V)
        fi = np.imag(V)

        Key = np.concatenate((typekey[1],typekey[2]))
        Key=np.sort(Key)

        # 计算F(x)的值
        # P_delta = Psp - (ai[Key]*ei[Key] + bi[Key]*fi[Key]) 
        # Q_delta = Qsp - (ai[typekey[2]]*fi[typekey[2]] - bi[typekey[2]]*ei[typekey[2]])
        # VV_delta = V[typekey[1]]**2-ei[typekey[1]]**2-fi[typekey[1]]**2

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
    
    # 新增一条元数据——改变拓扑结构
    def AppendData(self,type,data):
        if type == 'bus':
            self.bus = self.bus.drop(data)
        elif type == 'branch':
            self.branch = self.branch.drop(data)
        elif type == 'gen':
            self.gen = self.gen.drop(data)
        pass

    # 删除一条元数据——改变拓扑结构
    def DeleteData(self,type,data):
        if type == 'bus':
            self.bus = self.bus.drop(data)
        elif type == 'branch':
            self.branch = self.branch.drop(data)
        elif type == 'gen':
            self.gen = self.gen.drop(data)
            
        pass
      # 运行牛拉法潮流
    def RunPF(self,el = 1e-6,lmax = 100):
        self._Get_Ybus()
        # self.Ybus = np.loadtxt("Ybus.txt",dtype=complex)
        typekey = self._bus_types()  # [ref,pv,pq]
        # init value

        G = self.Ybus.real
        B = self.Ybus.imag

        self.V0 = self.bus['Vm'] * np.exp(1j * np.pi / 180 * self.bus['Va'])
        self.V_nr = self.V0
        S_Gen,S_load = self._make_Sin()
        # NR begin!!
        lnum = 0
        while True: # 最多100次迭代
            lnum = lnum + 1 # 迭代次数加1
            # x_k+1 = x_k - J(x_k)^-1 * F(x_k)
            Fx,ai,bi,ei,fi = self._make_fx(self.V_nr,typekey)
            Jac = self._make_Jac(ai,bi,ei,fi,G,B,typekey)
            delta = np.linalg.solve(Jac,-Fx)
            self.eifi = self._V2eifi(self.V_nr,typekey)
            self.eifi = self.eifi + delta
            print("max",np.max(np.abs(Fx)))
            self.V_nr = self._eifi2V(self.eifi,typekey)
            if np.max(np.abs(Fx)) < el:
                self._ProcessRes(S_Gen,S_load,typekey)
                return "NR converge"
            if lnum > lmax:
                self._ProcessRes(S_Gen,S_load,typekey)
                return "[warn]:Plase Mind !!!! The N-R is not converge!!"
    

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
    
if __name__ == "__main__": # 这里是使用的例子
    a = PowerFlow("case300.mat")
    a.RunPF()
    Res2Excel_bus(a, "result.xlsx", "result_bus")
    df = pd.DataFrame(np.round(a.Ybus,3) )
    df.to_excel("Ybus.xlsx", sheet_name="Ybus")
    print("cost:",a.Cost)