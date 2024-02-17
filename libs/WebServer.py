import threading
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.gen
import numpy as np
import json
import cv2
import base64
import time
import signal

class WebServer:
    clients = set()

    def __init__(self,configDict):
        self.id = id(self)
        global config
        config = configDict        
        self.app = tornado.web.Application([
            (r"/", MainHandler),
            (r"/ws", WebSocketHandler),
        ], static_path='www/static', template_path='www/templates')
        self.data = np.random.rand(100, 100, 3) * 255
        self.data_event = threading.Event()
        self.exit_signal = threading.Event()
        self.periodic_callback = tornado.ioloop.PeriodicCallback(self.update_clients, 100)  # Call every 100 milliseconds
        self.thread = None

    def start(self,*args,**kwargs):
        self.thread = threading.Thread(target=self.run_server)
        self.thread.start()

    def stop(self,*args,**kwargs):
        print("Stopping server...")
        self.periodic_callback.stop()
        self.exit_signal.set()

    def stop_server(self):
        self.app.stop()  # Stop accepting new connections
        io_loop = tornado.ioloop.IOLoop.current()
        io_loop.add_timeout(time.time(), io_loop.stop) 

    def run_server(self):
        self.app.listen(config['HTTP_PORT'])
        print(f"Server started on http://localhost:{config['HTTP_PORT']}/")
        self.periodic_callback.start()
        while not self.exit_signal.is_set():
            tornado.ioloop.IOLoop.current().start()

    def update(self, *args):
        self.data = args if len(args) == 1 else list(args)
        self.data_event.set()

    @tornado.gen.coroutine
    def update_clients(self):
        if self.data_event.is_set():
            datas = { "imgs": [ self.perform_image_processing(data) for data in self.data ] }
            for handler in WebServer.clients:                
                try:
                    yield handler.write_message(datas)
                except Exception as e:
                    print(f"Error sending data to client {id(handler)}: {e}")
            self.data_event.clear()

    def perform_image_processing(self, image_np_array):
        _, img_encoded = cv2.imencode('.jpg', image_np_array)
        image_base64 = base64.b64encode(img_encoded).decode('utf-8')
        return image_base64

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", cfg=config)

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print(f"WebSocket opened for client {id(self)}")
        WebServer.clients.add(self)

    def on_message(self, message):
        pass  # You can implement specific actions when a message is received from the client

    def on_close(self):
        print(f"WebSocket closed for client {id(self)}")
        WebServer.clients.remove(self)

if __name__ == "__main__":
    server = WebServer()
    server.start()
    try:
        while True:
            img = np.random.rand(100, 100, 3) * 255
            server.update(img)
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
