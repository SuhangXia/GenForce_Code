from ur5e import *
import time



if __name__ == '__main__':


    sensor = "hetero/gelsight"  
    with open(f"components/gelsight/config/hetero_gelsight.yaml",'r') as fp:  
        config = yaml.safe_load(fp)  
    ctime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
    indenter = config["Indenter"]  
    dir_ft = f"data/{sensor}/force/{indenter}-{ctime}"  
    dir_imgs = f"data/{sensor}/image/{indenter}-{ctime}"  
    dir_pos = f"data/{sensor}/pose/{indenter}-{ctime}"  
    dir_path = f'data/{sensor}/path/{indenter}-{ctime}'  
    os.makedirs(dir_ft, exist_ok=True)  
    os.makedirs(dir_imgs, exist_ok=True)  
    os.makedirs(dir_pos, exist_ok=True)  
    os.makedirs(dir_path, exist_ok=True)  

    rob = init_robo(config['ip'])
    allPoints = relativePath(rob.getl(),**config['path'])
    np.savetxt(f'{dir_path}/{ctime}.csv',allPoints.reshape(-1,6),delimiter=',',fmt="%f")

    port = config['GelSight']['port']
    gelsight = GelSight(port,dir_imgs)
    gelsight.c_ref()

    collect_calib = True
    collect_modulus = False

    with nidaqmx.Task() as task:

        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1")
        task.ai_channels.add_ai_voltage_chan("Dev1/ai2")
        task.ai_channels.add_ai_voltage_chan("Dev1/ai3")
        task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
        task.ai_channels.add_ai_voltage_chan("Dev1/ai5")
        ft = FT_Sensor(task,dir_ft)

        t_start = time.time()
        if collect_calib:   
            robo_run(port,ft,rob,allPoints,dir_ft,dir_imgs,dir_pos,**config['robo'])
        elif collect_modulus:
            robo_run_modulus(port,ft,rob,config["modulus"]["depth"],dir_ft,dir_imgs,dir_pos,ban_img=True,**config['robo'])
        t_end = time.time()
        t_cost = abs(t_start-t_end)/60
        print(f"finished: {t_cost} mins")



