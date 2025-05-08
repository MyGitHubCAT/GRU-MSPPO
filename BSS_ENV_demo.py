#env demo for simualtion
#any questions, welcome further discussion.(405648708@sjtu.edu.cn)
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from collections import deque

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Battery(object):
    def __init__(self,id: int, cpt, btransit_step=1,bstageduration = 1):
        self.step_duration = 1 
        self.SOC = 1
        self.capacity = cpt 
        self.ID = id
        self.if_charge_end = True 
        self.charge_type = 1 # 
        self.charge_target = 1 
        self.charge_stage = 1 
        self.SOC_stage1 = 100/120 #
        self.SOC_stage2 = 1
        self.chargingpower1 = 100 
        self.chargingpower2 = 20        
        self.request_total_energy = 0
        self.request_stage_energy = 0
        self.chargingSOC = 0 
        self.charging_request_time = 0 
        self.charging_total_time = 0 
        self.transit_k = btransit_step #total transit time
        self.ctransit = 0 #Time in transit
   
    def ChargePlanStart(self, full_charge : int):# 1 for full, 0 for 80% charge(0 is ignored in this demo)
        self.charge_stage = 1 #
        self.request_total_energy = 0
        self.request_stage_energy = 0#
        self.charge_type = full_charge
        self.charge_target = self.SOC_stage2 if full_charge == 1 else self.SOC_stage1
        self.if_charge_end = False
        self.request_total_energy = self.capacity * self.charge_target
        self.request_stage_energy  = self.capacity * self.SOC_stage1

    def Charge(self):
        if self.if_charge_end:
            return
        message1 = 'Battery{0}, need total soc{1},stage soc{2}\n'.format(self.ID,self.request_total_energy,self.request_stage_energy)
        if self.charge_stage == 1:
            self.SOC = self.SOC_stage1
            self.request_stage_energy = self.request_total_energy- self.request_stage_energy
            
        if abs(self.charge_stage - 2) < 0.001:
            self.SOC = self.SOC_stage2
            self.request_stage_energy = 0

        self.charge_stage += 1
        if abs(self.request_stage_energy-0)<0.001 :
                self.FinishCharging()
        message2 = 'battery{0} stage charged,soc{1},targetSOC{2},if end{3}'.format(self.ID,self.SOC,self.charge_target,self.if_charge_end)
        return message1+message2       
    def FinishCharging(self):
        self.charge_stage = 1
        self.if_charge_end = True
    def SOCChange(self,changesoc):
        self.SOC = round(max(0,self.SOC-changesoc),4)
        message = 'battery{0},soc{1}'.format(self.ID,self.SOC)
        return message          

    def Message(self,if_print= False):
        
        message = 'battery{0},SOC{1}'.format(self.ID,self.SOC)
        if if_print:
            print('battery{0}，capacity{1},'.format(self.ID,self.capacity) +
                'SOC{0}\n'.format(self.SOC))
        return message
 
class ElectricVehicle(object):
    def __init__(self, id, cpt = 120,evtransit_step = 1,evstageduration = 1):
        self.cpt = cpt
        self.evtransitstep = evtransit_step 
        self.stageduration = evstageduration
        self.battery = Battery(id,cpt,btransit_step = evtransit_step,bstageduration= evstageduration)
        self.id = id
        self.workstate=0
        self.energy_consumption = 30 
        self.swap_soc =0 

        self.used_soc = [min((id//2+1),5)*0.1,0.1]
    
    def Reinitial(self):
        self.battery = Battery(self.id,self.cpt,self.evtransitstep,self.stageduration)
        self.workstate=0 #0 work 1 stop

    def NeedSwap(self):
        needswap = True if abs(self.battery.SOC - self.swap_soc)<0.01 else False
        return needswap
    
    def Update(self,working_soc = None, ratio = 1):
        stage_used_soc =  ratio *(self.used_soc[0] if self.workstate == 0 else self.used_soc[1])
        if working_soc != None:
            stage_used_soc = ratio * (random.randrange(6,8)*working_soc/10 + (stage_used_soc*random.randrange(1,4) / 100))
        needswap,workingmessage = self.Working(stage_used_soc)
        return needswap,workingmessage
        
    def Working(self,usedsoc):
        message1 = 'EV{0} in working state, soc consume{1}, battery inf:\n'.format(self.id,usedsoc)
        bmessage = self.battery.SOCChange(usedsoc)
        needswap = self.NeedSwap()
        message2 = '\nif need swap: {0}'.format(needswap)
        message = message1+bmessage+message2
        return needswap,message
    
    def Message(self):
        print('EV{0}. STATE{1}, soc consume{2}。'.format(self.id,self.workstate,self.used_soc[self.workstate]))
        print('battery inf ') 
        self.battery.Message()   

class BSS(object):
    def __init__(self,id_start, SCALE: int = 20, cpt = 120, bsstransit_step = 1 ,charger_num = 5,stageduration = 1):
        self.bsstransitstep = bsstransit_step 
        self.stage_duration = stageduration
        self.T_BNUM = SCALE
        self.charger_NUM = charger_num#
        self.charging_power = 100 #
        #full charged batteries
        self.BatteryT1_wait = deque()
        self.BatteryT1_available = deque()
        self.BatteryT1_quit = deque()
       
        #80 percentage charged batteries #ignored setting
        self.BatteryT2_wait = deque()
        self.BatteryT2_available = deque()
        self.BatteryT2_quit = deque()
        
        #swapped batteries
        self.BatteryT3_wait = deque()#
        self.BatteryT3_available = deque()#
        self.BatteryT3_plan = deque()#
        self.BatteryT3_charging = deque()#
        self.EV_wait = deque()
        self.EV_available = deque()
        self.EV_quit = deque()
        self.Initial(id_start, cpt)
        self.plan_charge_count = 0
        self.stage_charge_count = 0
        self.out_count = 0 
        self.MAX_energy_request = self.charging_power * self.charger_NUM
        self.energy_request = 0
        self.ps_energy = 0
        self.pvps_energy = 0
        self.PV_energy_ratio = 0#
        self.stage_energy_get = 0#
        self.stage_energy_out=0 #
        self.stage_carbon = 0
        self.stage_expense = 0
        self.carbon_cost = 0
        self.energy_cost = 0
        self.stage_swapped_ev_num = 0
        self.stage_unswapped_ev_num = 0
        self.mask = np.zeros((self.T_BNUM))
        self.BT1_readdy_num = 0#
        self.BT2_readdy_num = 0#
        self.BT3_readdy_num = 0#

    def Initial(self,ids,b_cpt):
        for i in range(self.T_BNUM):
            battery = Battery(i+ids, b_cpt,self.bsstransitstep,self.stage_duration)
            self.BatteryT1_available.append(battery)
        self.stage_swapped_ev_num = 0

    def Update(self):
        self.BSSSwapAction()

    def BSSAddEV(self, ev):
        self.EV_wait.append(ev)

    def BSSSwapAction(self):#
        available_battery_num = len(self.BatteryT1_available)+len(self.BatteryT2_available)
        waiting_num = len(self.EV_wait)
        swap_num = min(waiting_num,available_battery_num)#
        T1_num = min(swap_num,len(self.BatteryT1_available))#
        T2_num = min(len(self.BatteryT2_available),swap_num-T1_num)#
        message1 = 'swapping request num{0},ava battery{1}, served ev{2}, T1 battery{3}\n'.format(
            waiting_num,available_battery_num, swap_num,T1_num)
        for i in range(T1_num):
            swap_batteryT1 = self.BatteryT1_available.popleft()
            swapped_ev = self.EV_wait.popleft()
            old_b = self.SwapBatteryforOneEV(swapped_ev,swap_batteryT1)

            self.BatteryT3_wait.append(old_b)
            self.EV_quit.append(swapped_ev)
        for i in range(T2_num):
            swap_batteryT2 = self.BatteryT2_available.popleft()
            swapped_ev = self.EV_wait.popleft()
            old_b = self.SwapBatteryforOneEV(swapped_ev,swap_batteryT2)
            
            self.BatteryT3_wait.append(old_b)
            self.EV_quit.append(swapped_ev)
        self.stage_swapped_ev_num = len(self.EV_quit)
        self.stage_unswapped_ev_num = waiting_num - self.stage_swapped_ev_num
        return message1
        
    def SwapBatteryforOneEV(self,ev: ElectricVehicle, avl_battery:Battery):
        old_battery = ev.battery
        ev.battery = avl_battery
        return old_battery
    
    def BatteriesChargingPlan(self,batteryplan= 1):#
        self.BatteryT3_Mask()#
        battery_plan =batteryplan * self.mask  
        for i in range(len(self.BatteryT3_available)):
            chargingplan_b = self.BatteryT3_available.popleft()
            chargingplan_b.ChargePlanStart(battery_plan[i])
            self.BatteryT3_plan.append(chargingplan_b)
        new_Cplan = battery_plan[:int(sum(self.mask))] 
        current_Cplan = [ i.charge_type for i in self.BatteryT3_plan]
        message1 = 'charging plan{0},curt cp{1}'.format(new_Cplan,current_Cplan)
        return message1

    def BatteryT3_Mask(self):
        self.mask = self.mask * 0
        for i in range(len(self.BatteryT3_available)):
            self.mask[i] = 1

    def Chargercount(self,chargecount=5,pvratio=0):
        self.plan_charge_count = min(chargecount,self.charger_NUM)   
        self.PV_energy_ratio = pvratio

    def BatterieschargingbycountOneStage(self,pstation,pvpstation,maxpvpsenergy):
        charging_inf = []#
        finish_charge_count = 0
        finish_t1charge_count = 0
        finish_t2charge_count = 0
        plan_count = len(self.BatteryT3_plan)#
        self.stage_charge_count = min(plan_count,self.plan_charge_count)
        message1 = 'working charging bays num'.format(self.stage_charge_count)
        charging_inf.append(message1)

        self.stage_energy_get=0#
        self.out_count = self.plan_charge_count - self.stage_charge_count

        for i in range(self.stage_charge_count):
            message2 = 'charging count {0}，cp {1}，consume energy{2}'.format(i, self.BatteryT3_plan[0].charge_target,self.BatteryT3_plan[0].request_stage_energy)
            charging_inf.append(message2)
           
            b_plan = self.BatteryT3_plan.popleft()
            self.stage_energy_get += b_plan.request_stage_energy
             
            message3 = b_plan.Charge()
            charging_inf.append(message3)
            #
            if b_plan.if_charge_end:
                finish_charge_count += 1
                if abs(b_plan.charge_type - 1)<0.0001:
                    finish_t1charge_count += 1
                    self.BatteryT1_wait.append(b_plan)
                if abs(b_plan.charge_type - 0)<0.0001:
                    finish_t1charge_count += 1
                    self.BatteryT2_wait.append(b_plan)
            else:
                self.BatteryT3_charging.append(b_plan)
            
        for i in range(len(self.BatteryT3_charging)):
            b_still_need_charging = self.BatteryT3_charging.pop()
            self.BatteryT3_plan.appendleft(b_still_need_charging)

        self.energy_request = self.stage_energy_get
        self.stage_energy_out = 0

        message5='request charging bays num{0}, actual num{1}, need battery num {2}. \n  total consume{3}, end num{4}'.format(
            self.plan_charge_count,self.stage_charge_count, plan_count,self.stage_energy_get ,
            finish_charge_count)
        charging_inf.append(message5)

        getenergymessage = self.GetEnergy(pstation,pvpstation,maxpvpsenergy)
        charging_inf.append(getenergymessage)

        return charging_inf
    def BatterieschargingOneStage(self,pstation,pvpstation,maxpvpsenergy):
        pass

    def EnergyRequest(self,energyrequest=100,pvratio=0):
        self.energy_request = min(energyrequest,self.MAX_energy_request) 
        self.PV_energy_ratio=pvratio
        self.stage_energy_get = self.ps_energy + self.pvps_energy#

    def GetEnergy(self, pstation,pvpstation,maxpvpsenergy):
        pvps_energy_request = self.energy_request * self.PV_energy_ratio
        
        self.pvps_energy = min(maxpvpsenergy,pvps_energy_request)

        self.ps_energy = max(self.energy_request - self.pvps_energy,0)              

        pscarb,psexpense,psmessage = self.CostInOnestageInPS(self.ps_energy,pstation)
        
        pvpscarb,pvpsexpense,pvpsmessage = self.CostInOnestageInPS(self.pvps_energy,pvpstation)

        self.stage_carbon = pscarb+pvpscarb
        self.stage_expense = psexpense+pvpsexpense
        message1= 'consume energy{0},pv ratio{1},pv{2},grid{3},carbon {4}, energy price{5}'.format(
                    self.energy_request,self.PV_energy_ratio,self.pvps_energy,self.ps_energy,self.stage_carbon,self.stage_expense)
        return message1 + psmessage + pvpsmessage
    
    def CostInOnestageInPS(self,energyrequest,powerstation):
        carb,expense,message = powerstation.ChargeCost(energyrequest)
        return carb,expense,message  

    def FreshWaitBattery(self):
        self.BT1_readdy_num = len(self.BatteryT1_wait)
        self.BT2_readdy_num = len(self.BatteryT2_wait)
        self.BT3_readdy_num = len(self.BatteryT3_wait)
        
        for i in range(self.BT1_readdy_num):
            BT1_readdy = self.BatteryT1_wait.popleft()
            if BT1_readdy.ctransit >= BT1_readdy.transit_k:
                self.BatteryT1_available.append(BT1_readdy)
                BT1_readdy.ctransit = 0
            else:
                BT1_readdy.ctransit += 1
                self.BatteryT1_wait.append(BT1_readdy)

        for i in range(self.BT2_readdy_num):
            BT2_readdy = self.BatteryT2_wait.popleft()
            if BT2_readdy.ctransit >= BT2_readdy.transit_k:
                self.BatteryT2_available.append(BT2_readdy)
                BT2_readdy.ctransit = 0
            else:
                BT2_readdy.ctransit += 1
                self.BatteryT2_wait.append(BT2_readdy)

        for i in range(self.BT3_readdy_num):
            BT3_readdy = self.BatteryT3_wait.popleft()
            if BT3_readdy.ctransit >= BT3_readdy.transit_k:
                self.BatteryT3_available.append(BT3_readdy)
                BT3_readdy.ctransit = 0
            else:
                BT3_readdy.ctransit += 1
                self.BatteryT3_wait.append(BT3_readdy)
           
    def MessageChargingBattertplan(self):
        print('charging plan b nums{0}'.format(len(self.BatteryT3_plan)))
        for i in range(len(self.BatteryT3_plan)):
            print('battery id{0}, soc{1}, cp{2},charge stage{3}'.format(self.BatteryT3_plan[i].ID,
                                                                self.BatteryT3_plan[i].SOC,
                                                                self.BatteryT3_plan[i].charge_target,
                                                                self.BatteryT3_plan[i].charge_stage))
    def MessageWaitingPlanBattery(self):
        print('waiting arrange cp b num{0}'.format(len(self.BatteryT3_available)))
        for i in range(len(self.BatteryT3_available)):
            print('battery id{0}, soc{1}'.format(self.BatteryT3_available[i].ID,self.BatteryT3_available[i].SOC))
    def Message(self):
        print('BSS inf:')
        print('charging plan b num{0}'.format(len(self.BatteryT3_plan)))      

class PS(object): # grid average CO2 factor 0.5703t/MWh->570.3g/kWh,pv  average CO2 factor 2-81g/kWh(set 40 round in this study)
    def __init__(self, carbonfactor = 570.3 ,MAX_ENERGY_PERSTAGE = 99999):
        self.ptype = 0 #
        self.carbon_factor = carbonfactor 
        self.stage_energy = MAX_ENERGY_PERSTAGE #
        self.price = 0
        
    def ChangePrice(self,cprice):
        self.price = cprice

    def ChargeCost(self,usedenergy):
        carb = self.CarbonEmission(usedenergy)
        expense = self.EnergyExpense(usedenergy)
        typename = 'grid' if self.ptype == 0 else 'pv'
        message = '\n energy from {0}.co2 factor{1} g/kWh.energy price{2} $/kwh. usedenergy{3}. co2{4}, expense{5}'.format(
             typename,self.carbon_factor,self.price,usedenergy, carb,expense)
        return carb,expense,message
    def CarbonEmission(self,usedenergy):#
        carbon = usedenergy * self.carbon_factor
        return carbon
    def EnergyExpense(self,usedenergy):#
        expense = usedenergy *self.price
        return expense
    
    def Message(self):
        print('energy from{0}.\n co2 factor{1} g/kWh.\n usedenergy{2}.\n price{3} $/kwh.'.format(
            'grid' if self.ptype == 0 else 'pv',self.carbon_factor,self.stage_energy,self.price))
        
class PVPS(PS):
    def __init__(self, carbonfactor = 40, init_stage_energy = 1000):
        super(PVPS,self).__init__(carbonfactor, MAX_ENERGY_PERSTAGE = init_stage_energy)
        self.ptype = 1
    def ChangeEnergy(self,eg):#
        self.stage_energy = eg
    

class GameManager(object):
    def __init__(self,evnum,chargernum=5, default_energydata_length = 300, max_pv_eg = 200, pvprice=20, transit_step=1,
    adenergydatanum = 24,ps_eprice_ds = None,pvps_eprice_ds = None,pvps_se_ds = None):
        self.EV_NUM = evnum
        self.EV_fleet = deque()
        self.EV_fleet_work = deque()
        self.fleet_ev_num = evnum
        self.bsschargernum = chargernum
        self.bss = BSS(self.EV_NUM,evnum,bsstransit_step=transit_step,charger_num=chargernum ) 
        self.ps = PS()
        self.pvps = PVPS()   
        self.max_pveg = max_pv_eg
        self.pv_price = pvprice
        self.transitstep = transit_step
        self.time = 0
        self.time_24 = 0
        self.wait_evnum = 0
        self.preper_evnum = 0
        self.ava_bnum = 0 
        self.ava_bt1num = 0 
        self.aba_bt2num = 0 
        self.stage_request_chargers = 0 
        self.stage_work_chargers = 0 
        self.stage_used_energy = 0
        self.stage_used_gdenergy = 0
        self.stage_used_pvenergy = 0

        self.ps_energy_price = []
        self.pvps_energy_price = []
  
        self.pvps_stage_energy = []

        self.ev_worksoc_data = []

        self.energydata_length = default_energydata_length 
        self.adenergydata_num = adenergydatanum 
        self.ed_length = self.energydata_length + adenergydatanum+1 
        
        self.Update_Stage_Inf = [] 
        self.Initial(ps_eprice_ds,pvps_eprice_ds,pvps_se_ds, False)
        
    def Initial(self,ps_eprice_ds = None,pvps_eprice_ds = None,
                pvps_senergy_ds = None, reinitial: bool = True):
        self.EV_fleet.clear()    
        for i in range(self.EV_NUM):
            init_ev = ElectricVehicle(i,evtransit_step=self.transitstep) # 
            self.EV_fleet.append(init_ev)
        self.fleet_ev_num = self.EV_NUM
        self.bss = BSS(self.EV_NUM,self.EV_NUM,bsstransit_step=self.transitstep,charger_num=self.bsschargernum)        
        self.ps = PS()
        self.pvps = PVPS()    
        self.time = 0
        self.time_24 = 0
        self.is_done = 0
        self.wait_evnum = 0
        self.preper_evnum = 0
        self.UpdateBSSInf()
        self.Update_Stage_Inf.clear()
        if not reinitial:
            self.InitialEnergyDataset(ps_eprice_ds,pvps_eprice_ds,pvps_senergy_ds)  
        self.FreshEnergyData()
    def InitialEnergyDataset(self,ps_ep_ds = None,pvps_ep_ds = None,pvps_se_ds=None):   
        self.ps_energy_price.clear()
        self.pvps_energy_price.clear() 
        self.pvps_stage_energy.clear()
        self.ev_worksoc_data.clear()
        if (type(ps_ep_ds) != type(None)) and (len(ps_ep_ds)>= self.ed_length):
            for i in ps_ep_ds:
                self.ps_energy_price.append(i)
        else:           
            for i in range(self.ed_length):
                self.ps_energy_price.append(4+8*math.sin((-math.pi) + (((i-24)*math.pi)/24)))
        if (type(pvps_ep_ds) != type(None)) and (len(pvps_ep_ds)>= self.ed_length):
            for i in pvps_ep_ds:
                self.pvps_energy_price.append(i)
        else:           
            for i in range(self.ed_length):
                self.pvps_energy_price.append(self.pv_price)
        if (type(pvps_se_ds) != type(None)) and (len(pvps_se_ds)>= self.ed_length):
            for i in pvps_se_ds:
                self.pvps_stage_energy.append(i)
        else:
            for i in range(self.ed_length):
                self.pvps_stage_energy.append(self.max_pveg)
        
        for i in range(self.energydata_length+1):
            worksoc = 120* self.norm_data(i,130,55) if 120* self.norm_data(i,130,55)> 0.01 else 0
            self.ev_worksoc_data.append(worksoc) 
        
    def norm_data(self, i,mu,sigma ):
        return math.exp(-(math.pow((i-mu)/sigma,2)/2))/(sigma*math.sqrt(2*math.pi)) 

    def FreshEnergyData(self):
        self.ps.ChangePrice(self.ps_energy_price[self.time + self.adenergydata_num])
        self.pvps.ChangePrice(self.pvps_energy_price[self.time + self.adenergydata_num])
        self.pvps.ChangeEnergy(self.pvps_stage_energy[self.time + self.adenergydata_num])     

    def EnvState(self):
        ava_bt1_num = len(self.bss.BatteryT1_available)
        ava_bt2_num = len(self.bss.BatteryT2_available)
        charge_count = self.bss.BT2_readdy_num    
        ctime = self.time
        old_price = self.ps_energy_price[self.time: self.time + self.adenergydata_num]
        cost = self.bss.stage_expense
        carbon = self.bss.stage_carbon
        wait_ev_length = self.bss.stage_unswapped_ev_num
        wasted_count = self.bss.out_count
        is_done = self.is_done
                
        return np.array([ava_bt1_num,ava_bt2_num,wait_ev_length,ctime]),old_price ,np.array([cost,carbon,wait_ev_length,wasted_count,charge_count]),is_done    

    def Update(self,energy_act,pv_ratio_act=0, chargeplan = 1,type = True):
        self.Update_Stage_Inf.clear()
        self.UpdateEnvdata()#
        self.UpdateSwapEVBSS()#
        self.UpdateBSSPlan(chargeplan)#
        self.UpdateBSSChargingByCount(energy_act,pv_ratio_act)
        self.UpdateEVFleet()#
        self.UpdateEnvInf()#

    def UpdateEnvdata(self):
        
        self.time += 1
        self.time_24 = self.time % 24
        self.FreshEnergyData()
        
    def UpdateSwapEVBSS(self):
        message1 = 'EV BSS SWAP'
        self.Update_Stage_Inf.append(message1)
        
        swap_message = self.bss.BSSSwapAction()
        self.Update_Stage_Inf.append(swap_message)

        for i in range(self.bss.stage_swapped_ev_num):
            swaped_ev = self.bss.EV_quit.popleft()
            self.EV_fleet.append(swaped_ev)
        message2 = 'EV BSS SWAP end'
        self.Update_Stage_Inf.append(message2)

    def UpdateBSSPlan(self,BT3_ava_chargingplan=1):#
        message1 = self.bss.BatteriesChargingPlan(BT3_ava_chargingplan)
        self.Update_Stage_Inf.append(message1)
        #
    def UpdateBSSChargingByCount(self,stagechargecount=5,stagepvratio=0):
        message1 = 'charging'
        self.bss.Chargercount(stagechargecount,stagepvratio)
        message2_list = self.bss.BatterieschargingbycountOneStage(self.ps,self.pvps,self.pvps.stage_energy)
        #
        self.Update_Stage_Inf.append(message1)
        self.Update_Stage_Inf += message2_list

    def UpdateBSSCharging(self,stageengeryrequest=400,stagepvratio=0):
        message1 = 'charging'
        self.bss.EnergyRequest(stageengeryrequest,stagepvratio) 
        message2_list = self.bss.BatterieschargingOneStage(self.ps,self.pvps,self.pvps.stage_energy)
        #
        self.Update_Stage_Inf.append(message1)
        self.Update_Stage_Inf += message2_list
        #self.bss.MessageChargingBattertplan()      

    def UpdateEVFleet(self):
        self.wait_evnum = len(self.bss.EV_wait)#
        self.fleet_ev_num= len(self.EV_fleet)
        ccount = 0
        message1 = 'EV fleet：'
        self.Update_Stage_Inf.append(message1)

        for i in range(self.fleet_ev_num):
            ev = self.EV_fleet.popleft()
            timeratio = self.time 
            needswap,ev_message = ev.Update(self.ev_worksoc_data[self.time],ratio = 1)
            self.Update_Stage_Inf.append(ev_message)
            if needswap:
                self.bss.BSSAddEV(ev)
                ccount+=1
            else:
                self.EV_fleet.append(ev)
        self.preper_evnum = len(self.bss.EV_wait) #
        message2 = '{0}EV waiting swapping，{1}EV working in next stage'.format(ccount,self.fleet_ev_num-ccount)
        self.Update_Stage_Inf.append(message2)
        
    def UpdateEnvInf(self):
        self.bss.FreshWaitBattery()#
        self.is_done = 0 if self.time <= self.energydata_length -1 else 1
        self.UpdateBSSInf()
        if self.is_done:
            message = 'game over'+ str(self.energydata_length)
            self.Update_Stage_Inf.append(message)
        return 
    #
    def UpdateBSSInf(self):
        self.ava_bt1num = len(self.bss.BatteryT1_available)
        self.aba_bt2num = len(self.bss.BatteryT2_available)
        self.ava_bnum = self.ava_bt1num + self.aba_bt2num 

        self.stage_request_chargers = self.bss.plan_charge_count
        self.stage_work_chargers = self.bss.stage_charge_count

        self.stage_used_energy = self.bss.energy_request
        self.stage_used_pvenergy = self.bss.pvps_energy
        self.stage_used_gdenergy = self.bss.ps_energy

    def Message(self):
        print('stage {0} env inf'.format(self.time))
        for i in self.Update_Stage_Inf:
            print(i)       
        print('stage end, try swapping ev{0}, already request swapping{1}'.format(self.preper_evnum-self.wait_evnum,self.wait_evnum))
        print('in bss, available batteries {0}'.format(len(self.bss.BatteryT1_available)))   
        print('\n')  
       