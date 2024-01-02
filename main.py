from sensors.sensors import SensorEnv
from model.VisionModel_Integrated import FootDetector as integrated_model
from sensors.app.FramerateMonitor import FramerateMonitor
from unreal.env import DogChasingEnv
import numpy as np
import cv2
import time
import os
import pickle
import datetime


def visualize(image):
    if image.dtype != np.uint8:
        image *= 255
        image[image < 0] = 0
        image = image.astype(np.uint8)
    image = cv2.resize(image, (500, 500))
    cv2.imshow("Pressure", image)
    if cv2.waitKey(1) & 0xff == 27:
        return False
    else:
        return True

def init_sensor():
    print("initializing sensors...")
    sensor = SensorEnv(
        ports=["COM10", "COM12", 'COM8', 'COM11'], # for 4 sensors
        # ports=["COM10"], # for 1 sensor
        stack_num=20, # was 20 / 2022.09.28
        adaptive_calibration=True,
        normalize=True
    )
    print("sensor init finish")
    return sensor

def test_sensor(): # To test sensor input
    sensor = init_sensor()
    while True:
        images = sensor.get()
        if not visualize(images[-1]):
            break
        print(f"sensor FPS : {sensor.fps}")
    sensor.close()

def test_model(save_log = True, **kwargs): # To test model accuracy, collect data from diff angle and speed
    model = integrated_model(visualize=True)
    sensor = init_sensor()
    print('participant: ')
    name = input()
    print("beat per minute: ")
    bpm = input()
    sensing_count = 0

    if save_log:
        now = datetime.datetime.now()
        log_dir = os.path.join(
            kwargs['log_dir'],
            f"{now.month}_{now.day}_{name}"
        )
        os.makedirs(log_dir, exist_ok=True)

    try:
        while True:
            sensing_count += 1
            print(sensing_count)
            images = sensor.get()
            avail, angle, speed = model(images, hmd_yaw=0)

            visual_image = images[-1]
            if hasattr(model, "visualized_image"):
                visual_image = model.visualized_image
            if not visualize(visual_image):
                break
            if avail:
                angle = round(angle, 2)
                speed = round(speed, 2)
            print(f"sensor FPS:{sensor.fps}, Avail:{avail}, Angle:{angle}, Speed:{speed}")

            if save_log:
                if sensing_count == 1:
                    img_data = images[-1].reshape(64, 64, 1)
                    # img_data = images[-1].reshape(32, 32, 1)
                    data = np.array([sensing_count, speed, angle, time.time()]).reshape(4, 1)

                else:
                    img_data = np.concatenate((img_data, images[-1].reshape(64, 64, 1)), axis=2)
                    # img_data = np.concatenate((img_data, images[-1].reshape(32, 32, 1)), axis=2)
                    data = np.concatenate((data, np.array([sensing_count, speed, angle, time.time()]).reshape(4, 1)), axis=1)

    except:
        print("keyboard interrupted")
    finally:
        sensor.close()
        if save_log:
            print("Finish!")
            np.save(log_dir + '/' + str(bpm) + '_img', img_data)
            with open(log_dir + '/' + str(bpm) + '_data.pickle', 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def main(save_log=False, **kwargs): # main implementation
    env = DogChasingEnv("127.0.0.1", 13000) # connect to env (Unreal engine)
    fpsMonitor = FramerateMonitor()
    model = integrated_model(visualize=True)
    sensor = init_sensor()

    try:
        if save_log:
            now = datetime.datetime.now()
            log_dir = os.path.join(
                kwargs['log_dir'],
                f"{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
            )
            os.makedirs(log_dir, exist_ok=True)

            images_filename = "images.npy"
            infos_filename = "infos.pickle"
            fnp = open(os.path.join(log_dir, images_filename), 'wb')
            fpickle = open(os.path.join(log_dir, infos_filename), 'wb')

        while True:
            images = sensor.get()

            hmd_yaw = env.get_state()
            #hmd_yaw = 0

            avail, angle, speed = model(images, hmd_yaw=hmd_yaw)

            if avail:
                speed *= 1500 # adjust speed on game
                env.move(speed, angle, 1) # move given speed and given angle, forward

            visual_image = images[-1]
            if hasattr(model, "visualized_image"):
                visual_image = model.visualized_image # visualize model model output
            if not visualize(visual_image):
                break

            main_fps = round(fpsMonitor.getFps())
            sensor_fps = sensor.fps
            if avail:
                angle = round(angle, 2)
                speed = round(speed, 2)
            print(f"sensor FPS : {sensor_fps}, main FPS: {main_fps}, Speed: {speed}, Angle: {angle}, Avail: {avail}")
            fpsMonitor.tick()

            if save_log:
                np.save(fnp, images[-1])

                data = {
                    "sensor_fps": sensor_fps,
                    "main_fps": main_fps,
                    "avail": avail,
                    "speed": speed,
                    "angle": angle,
                    "hmd_yaw": hmd_yaw,
                    "time": time.time()
                }
                pickle.dump(data, fpickle, protocol=pickle.HIGHEST_PROTOCOL)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        sensor.close()
        if save_log:
            fnp.close()
            fpickle.close()

if __name__ == "__main__":
    main()
    # test_sensor()
    # test_model(save_log=True, log_dir=".\\logs")

