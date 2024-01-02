import sys
import socket
import time
import numpy as np


def sleep_decorator(func):
    def decorated(*args, **kwargs):
        time.sleep(0.0001)

        result = func(*args, **kwargs)
        return result
    return decorated

def bindUE4(ipaddr, port):
    address = (ipaddr, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(address)
    sock.listen(1)

    print("Connecting ... ")
    connectedSock, addr = sock.accept()
    print("Connected ... ")
    return connectedSock

@sleep_decorator
def GetAllState(connectedSock):
    connectedSock.sendall(bytes("GetState", "utf-8"))
    rowData = connectedSock.recv(300)
    recvMsg = rowData.decode("utf-8")
    result = recvMsg.split(",")

    return result

@sleep_decorator
def SensorWalk(connectedSock, speed, bodyDirection, axisX, axisY):
    command = ["SensorWalk", str(speed), str(bodyDirection), str(axisX), str(axisY)]
    command_str = " ".join(command) + " "
    connectedSock.sendall(bytes(command_str, "utf-8"))
