#!/home/pi/venv-3.9.2/bin/python
import time
from libs.TaskLooper import *
from libs.ImageTools import *
from libs.WebServer import *
import config

global cfg
cfg = config.__dict__
cfg = {key: value for key, value in cfg.items() if not key.startswith('__')}

# Exemple d'utilisation
if __name__ == "__main__":
    task_looper = TaskLooper(cfg)
    task_looper.add_task(captureImage(0,cfg), outputs=['img/1'])
    #task_looper.add_task(captureImage(1,cfg), outputs=['img/2'])
    task_looper.add_task(FaceDetector(cfg), inputs=['img/1'], outputs=['img/2'])    
    #task_looper.add_task(ObjectDetector(cfg), inputs=['img/1'], outputs=['img/2'])    
    task_looper.add_task(WebServer(cfg), inputs=['img/1','img/2'])
    # task_looper.add_task(saveImage(), inputs=['img/1'])

    # task_looper.add_task(capturImg(1), outputs=['img/2'])
    # task_looper.add_task(processImg(), inputs=['img/1'], outputs=['direction', 'acceleration'])
    # task_looper.add_task(processImg(), inputs=['img/2'], outputs=['light', 'objects'])
    task_looper.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting process ...")
        task_looper.stop()