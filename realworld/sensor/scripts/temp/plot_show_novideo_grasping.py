#!/usr/bin/env python3
import time
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import threading
import roslibpy
import argparse
import sys
from PyQt5.QtCore import QTimer

WINDOW_LENGTH = 10.0 

class DataStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.force1_x, self.force1_x_t = [], []
        self.force1_y, self.force1_y_t = [], []
        self.force1_z, self.force1_z_t = [], []
        self.force2_x, self.force2_x_t = [], []
        self.force2_y, self.force2_y_t = [], []
        self.force2_z, self.force2_z_t = [], []
        self.start_time = None

store = DataStore()

# ---- MODIFICATION: use an Event to signal exit ----
exit_event = threading.Event()

def ros_subscribe_thread(client, sensor1, sensor2):
    def make_cb(data_list, t_list):
        def cb(msg):
            t_now = time.time()
            with store.lock:
                if store.start_time is None:
                    store.start_time = t_now
                rel_t = t_now - store.start_time
                t_list.append(rel_t)
                data_list.append(msg['data'])

                while t_list and (rel_t - t_list[0]) > WINDOW_LENGTH:
                    t_list.pop(0)
                    data_list.pop(0)
        return cb

    topics = {
        f'/force/{sensor1}/x':    (store.force1_x, store.force1_x_t),
        f'/force/{sensor1}/y':    (store.force1_y, store.force1_y_t),
        f'/force/{sensor1}/z':    (store.force1_z, store.force1_z_t),
        f'/force/{sensor2}/x':    (store.force2_x, store.force2_x_t),
        f'/force/{sensor2}/y':    (store.force2_y, store.force2_y_t),
        f'/force/{sensor2}/z':    (store.force2_z, store.force2_z_t),
    }

    for topic_name, (data_list, t_list) in topics.items():
        topic = roslibpy.Topic(client, topic_name, 'std_msgs/Float32')
        topic.subscribe(make_cb(data_list, t_list))
        print(f"subscribed: {topic_name}")

    try:
        while client.is_connected:
            if exit_event.is_set():
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    exit_event.set()
    client.terminate()

def plot_window_main(sensor1,sensor2, client):
    app = QApplication([])
    win = pg.GraphicsLayoutWidget(title="Force Real-time Curves")
    win.resize(1000, 1400)

    pw1 = win.addPlot(title=f'{sensor1} FX / FY')
    pw1.setLabels(left='N', bottom='t(s)')
    curve1x = pw1.plot(pen=pg.mkPen('r', width=2), name=f'{sensor1} x')
    curve1y = pw1.plot(pen=pg.mkPen('g', width=2), name=f'{sensor1} y')

    win.nextRow()
    pw2 = win.addPlot(title=f'{sensor1} FZ')
    pw2.setLabels(left='N', bottom='t(s)')
    curve1z = pw2.plot(pen=pg.mkPen('b', width=2), name=f'{sensor1} z')

    win.nextRow()
    pw3 = win.addPlot(title=f'{sensor2} FX / FY')
    pw3.setLabels(left='N', bottom='t(s)')
    curve2x = pw3.plot(pen=pg.mkPen('m', width=2), name=f'{sensor2} x')
    curve2y = pw3.plot(pen=pg.mkPen('c', width=2), name=f'{sensor2} y')

    win.nextRow()
    pw4 = win.addPlot(title=f'{sensor2} FZ')
    pw4.setLabels(left='N', bottom='t(s)')
    curve2z = pw4.plot(pen=pg.mkPen('y', width=2), name=f'{sensor2} z')

    win.show()

    def update():
        with store.lock:
            curve1x.setData(store.force1_x_t, store.force1_x)
            curve1y.setData(store.force1_y_t, store.force1_y)
            curve1z.setData(store.force1_z_t, store.force1_z)
            curve2x.setData(store.force2_x_t, store.force2_x)
            curve2y.setData(store.force2_y_t, store.force2_y)
            curve2z.setData(store.force2_z_t, store.force2_z)
        # ---- MODIFICATION: If ROS client disconnected or exit event, quit app ----
        if (not client.is_connected) or exit_event.is_set():
            print("[INFO] ROS client disconnected or exit requested, closing window.")
            timer.stop()
            win.close()
            app.quit()

    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(30)
    try:
        app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        win.close()

def main():
    parser = argparse.ArgumentParser(description="Force Real-time Plot Script")
    parser.add_argument('--sensor1', required=True, help='Name of the first sensor (e.g. AII)')
    parser.add_argument('--sensor2', required=True, help='Name of the second sensor (e.g. uskin)')
    args = parser.parse_args()
    sensor1 = args.sensor1
    sensor2 = args.sensor2
    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    print("[roslibpy] Connected to rosbridge websocket.")
    t_ros = threading.Thread(target=ros_subscribe_thread, args=(client, sensor1, sensor2),daemon=True)
    t_ros.start()
    try:
        plot_window_main(sensor1,sensor2, client)
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt detected, shutting down...")
        exit_event.set()
    finally:
        client.terminate()
        print("[INFO] Clean exit.")

if __name__ == "__main__":
    main()