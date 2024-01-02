from unreal.TcpSender import *
import multiprocessing as mp
from multiprocessing import Manager

class DogChasingEnv:
    def __init__(self, ipaddr, port):
        self.ipaddr = ipaddr
        self.port = port
        self.connect()
        self.exit = mp.Event()
        self.body_angle_queue, self.axisX_queue, self.speed_queue, self.state_queue = Manager().Queue(), Manager().Queue(), Manager().Queue(), Manager().Queue()
        self.frame_queue = Manager().Queue()
        self.process = mp.Process(target=self._runner, args=())
        self.process.start()

    def connect(self):
        self.sock = bindUE4(self.ipaddr, self.port)


    def get_state(self):
        #allState = GetAllState(self.sock)
        #allState = self.state_queue.get()

        if self.state_queue.empty():
            allState = self.state_queue.get()
        else:
            while not self.state_queue.empty():
                allState = self.state_queue.get()


        hmd_yaw = allState[0]
        hmd_yaw = float(hmd_yaw)
        return hmd_yaw


    def move(self, speed, body_angle, axisX):
        self.speed = speed
        self.body_angle = body_angle
        self.axisX = axisX
        self.body_angle_queue.put(self.body_angle)
        self.speed_queue.put(self.speed)
        self.axisX_queue.put(self.axisX)

    def _runner(self):
        body_angle, speed, axisX = 0, 0, 0
        frame = np.zeros((64, 64))
        while not self.exit.is_set():
            if not self.body_angle_queue.empty():
                while not self.body_angle_queue.empty():
                    body_angle = self.body_angle_queue.get()

            if not self.speed_queue.empty():
                while not self.speed_queue.empty():
                    speed = self.speed_queue.get()

            if not self.axisX_queue.empty():
                while not self.axisX_queue.empty():
                    axisX = self.axisX_queue.get()

            if not self.frame_queue.empty():
                while not self.frame_queue.empty():
                    frame = self.frame_queue.get()
            SensorWalk(self.sock, speed, body_angle, axisX, 0)
            self.state_queue.put(GetAllState(self.sock))

    def close(self):
        self.exit.set()
        self.process.join()

def main():
    env = DogChasingEnv("172.27.186.198", 13331)
    print("Connected")

    for _ in range(10000):
        dog_location, player_location, distance, player_state, hmd_yaw = env.get_state()
        print("dog_location: {}, player_location: {}, distance: {}, player_state: {}, hmd_yaw: {}".format(
            dog_location, player_location, distance, player_state, hmd_yaw
        ))
        env.move(400, 270, 100)

if __name__ == "__main__":
    main()