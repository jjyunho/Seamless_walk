import serial
import numpy as np
import cv2
import multiprocessing as mp
from multiprocessing import Manager
import copy
from sensors.app.FramerateMonitor import FramerateMonitor

class Sensor:
    def __init__(self, port, baudrate, timeout):
        self.queue = Manager().Queue()
        self.exit = mp.Event()
        self.process = mp.Process(target=self._read2, args=(self.queue, port, baudrate, timeout))

    def start(self):
        self.process.start()
        #wait for init
    
    def close(self):
        self.exit.set()
        self.process.join()
    
    def get(self):
        result = None
        if self.queue.empty():
            result = self.queue.get()
        else:
            while not self.queue.empty():
                result = self.queue.get()
        return result
    
    def _read(self, queue, port, baudrate, timeout): # communicate with arduino board
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        i = 0
        while not self.exit.is_set():
            #read sensor values
            self.ser.reset_input_buffer()
            self.ser.write('a'.encode('utf-8'))
            w, h = 32, 32
            length = 2 * w * h
            input_string = self.ser.read(length)
            x = np.frombuffer(input_string, dtype=np.uint8).astype(np.uint16)
            
            #check initialization
            if len(x) != 2048:
                continue
            
            #reshape values
            x = x[0::2] * 32 + x[1::2]
            x = x.reshape(h, w).transpose(1, 0)

            #append queue
            queue.put(x)
            i += 1

    def _read2(self, queue, port, baudrate, timeout): # communicate with arduino board
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        _sensor_bitshift = 5
        _sensor_sample_size = (32, 32)

        i = 0
        while not self.exit.is_set():
            data = b''
            while len(data) == 0:
                self.ser.reset_input_buffer()
                self.ser.write('a'.encode('utf-8'))
                data = self.ser.readline()
            # Unpack the data.
            matrix_index = data[0] - (ord('\n') + 1)
            data_matrix = data[1:-1]  # ignore the newline character at the end
            data_matrix = np.frombuffer(data_matrix, dtype=np.uint8).astype(np.uint16)

            data_matrix = data_matrix - (ord('\n') + 1)
            data_matrix = data_matrix[0::2] * (2 ** _sensor_bitshift) + data_matrix[1::2]
            data_matrix = data_matrix.reshape(_sensor_sample_size)

            # append queue
            queue.put(data_matrix)
            i += 1

class MultiSensors:
    def __init__(self, ports):
        self.ports = ports
        self.make_sensors()

        self.queue = Manager().Queue()
        self.exit = mp.Event()
        self.process = mp.Process(target=self._read, args=(self.queue,))
        self.fps_monitor = FramerateMonitor()
        self.fps_queue = Manager().Queue()

    def make_sensors(self):
        sensors = []
        for port in self.ports:
            sensors.append(Sensor(port=port, baudrate=1000000, timeout=1.0))
        self.sensors = sensors
    
    def init_sensors(self):
        for sensor in self.sensors:
            sensor.start()

        init_values = []
        for sensor in self.sensors:
            x = sensor.get()
            init_values.append(x.astype(np.float32))

        init_values_num = 30
        for k in range(init_values_num):
            for i in range(len(self.sensors)):
                x = self.sensors[i].get()
                init_values[i] += x
        for i in range(len(self.sensors)):
            init_values[i] /= init_values_num
        self.init_values = init_values
        
        self.process.start()
    
    def _read(self, queue):
        while not self.exit.is_set():
            images = []
            for sensor in self.sensors:
                x = sensor.get()
                images.append(x)

            #concat
            if len(images) == 4:
                '''
                ========================================================================================================
                ====================This part should be modified if visualized image is not match=======================
                ========================================================================================================
                '''

                images[1] = cv2.rotate(images[1], cv2.ROTATE_90_CLOCKWISE)
                images[2] = cv2.rotate(images[2], cv2.ROTATE_90_COUNTERCLOCKWISE)
                images[2] = cv2.rotate(images[2], cv2.ROTATE_90_COUNTERCLOCKWISE)

                a = np.concatenate((images[1], images[0]))
                b = np.concatenate((images[3], images[2]))
                '''
                ========================================================================================================
                ========================================================================================================
                '''

                a = np.transpose(a, (1, 0))
                b = np.transpose(b, (1, 0))

                total_image = np.concatenate((a, b))
            else:
                images[0] = cv2.flip(images[0], 0)
                total_image = np.concatenate(images)
                # print("concat finish")
                # print(total_image)
            
            self.fps_monitor.tick()
            self.fps_queue.put(round(self.fps_monitor.getFps()))
            queue.put(total_image)
    
    def get(self):
        result = None
        if self.queue.empty():
            result = self.queue.get()
        else:
           while not self.queue.empty():
               result = self.queue.get()
        return result
    
    def get_all(self):
        if self.queue.empty():
            results = [self.queue.get()]
        else:
            results = []
            while not self.queue.empty():
                results.append(self.queue.get())
        return results

    def getFps(self):
        result = None
        if self.fps_queue.empty():
            result = self.fps_queue.get()
        else:
            while not self.fps_queue.empty():
                result = self.fps_queue.get()
        return result
    
    def close(self):
        self.exit.set()
        self.process.join()
        for sensor in self.sensors:
            sensor.close()

class SensorEnv:
    def __init__(self, ports, stack_num, adaptive_calibration, normalize=True):
        self.stack_num = stack_num
        self.normalize = normalize
        self.sensor = MultiSensors(ports)
        self.buffer = []

        denoise_sec = 1
        denoise_start = 2
        self.calibration_range = (
            int(-17 * (denoise_start+denoise_sec)),
            int(-17 * denoise_start)
        )
        assert stack_num < abs(self.calibration_range[1])
        self.adaptive_calibration = adaptive_calibration
        if adaptive_calibration:
            self.buffer_len = abs(self.calibration_range[0])
            self.calibration_step = self.buffer_len
        else:
            self.calibration_step = 200
            self.buffer_len = stack_num+1

        self.fps = 0
        self._ready()

    def _ready(self):
        self.sensor.init_sensors()
        while len(self.buffer) < self.calibration_step:
            self._read()
        base_value = self.buffer[-self.calibration_step:]
        base_value = np.array(base_value)
        self.base_value = base_value.mean(axis=0).astype('float32')

    def _read(self):
        self.buffer += self.sensor.get_all()
        if len(self.buffer) > self.buffer_len:
            self.buffer = self.buffer[-self.buffer_len:]

    def _preprocess(self, images):
        if self.adaptive_calibration:
            self.base_value = self.buffer[self.calibration_range[0]:self.calibration_range[1]]
            self.base_value = np.array(self.base_value).astype('float32')
            self.base_value = self.base_value.mean(axis=0)
        images -= np.expand_dims(self.base_value, axis=0)
        images /= 100
        images += 100
        return images

    def get(self):
        self._read()
        images = np.array(self.buffer[-self.stack_num:]).astype('float32')
        if self.normalize:
            images = self._preprocess(images)
        self.fps = self.sensor.getFps()
        return images

    def close(self):
        self.sensor.close()