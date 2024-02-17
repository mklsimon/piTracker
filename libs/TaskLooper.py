import time
import threading
import cv2

class TaskDefault:  
    def start(self):
        return True
    
    def stop(self):
        return True

class TaskLooper:
    def __init__(self, config):
        self.cfg = config
        self.interval = 1 / self.cfg['FREQ_LOOP_HZ']
        self.tasks = []
        self.shared_data = {}
        self.task_stats = {}
        self.loop_stats = []
        self.loop_count = 0
        self.loop_start = time.time()
        self.stats_table_header = ["Class", "Min (ms)", "Avg (ms)", "Max (ms)", "Total (sec)"]
        self.stats_table_format = "{:<20} {:<20} {:<20} {:<20} {:<20}"
        self.thread = threading.Thread(target=self._loop)
        self._stop_event = threading.Event()
        self.task_events = {}

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop_event.set()
        self.thread.join()
        for instance, inputs, outputs in self.tasks:
            instance.stop()
        self.print_task_stats()

    def add_task(self, task_instance, inputs=[], outputs=[]):
        self.task_stats[f"{type(task_instance).__name__}_{task_instance.id}"] = []
        for output in outputs:
            self.task_events[output] = threading.Event()
        print(f"load {type(task_instance).__name__}")
        task_instance.start()
        self.tasks.append((task_instance, inputs, outputs))

    def _loop(self):        
        while not self._stop_event.is_set():
            self.loop_count += 1
            loop_start_time = time.time()
            threads = []
            self.shared_data = {}

            # On lance les taches
            for task in self.tasks:
                instance, inputs, outputs = task
                thread = threading.Thread(target=self._execute_task, args=(instance, inputs, outputs))
                thread.start()
                threads.append(thread)

            # Attend que tous les threads se terminent avant de passer à la prochaine itération
            for thread in threads:
                thread.join()

            # Reinit des events
            for task in self.tasks:
                instance, inputs, outputs = task
                for output in outputs:
                    self.task_events[output].clear()

            self.loop_stats.append(time.time() - loop_start_time )
            sleep_time = max(0, (self.interval - (time.time() - loop_start_time)) )
            time.sleep(sleep_time)

    def _execute_task(self, task_instance, inputs, outputs):
        start_time = time.time()

        # Wait event to be ready
        for input in inputs:
            self.task_events[input].wait()

        inputs_data = {input_key: self.shared_data[input_key] for input_key in inputs}
        result = task_instance.update(*inputs_data.values())
        if outputs:
            result = [result] if len(outputs) == 1 else result
            for output_key, result_value in zip(outputs, result):
                self.shared_data[output_key] = result_value
                self.task_events[output_key].set()
        self.task_stats[f"{type(task_instance).__name__}_{task_instance.id}"].append(time.time() - start_time)

    def print_task_stats(self):
        def print_line_stats(function_name, execution_times):
            min_time = min(execution_times) * 1000  # convertir en millisecondes
            avg_time = (sum(execution_times) / len(execution_times)) * 1000  # convertir en millisecondes
            max_time = max(execution_times) * 1000  # convertir en millisecondes
            total_time = sum(execution_times)  # convertir en secondes
            row_data = [function_name, f"{min_time:.3f}", f"{avg_time:.3f}", f"{max_time:.3f}", f"{total_time:.3f}"]
            print(self.stats_table_format.format(*row_data))
        print("\n"+"-"*100)
        print(self.stats_table_format.format(*self.stats_table_header))
        print("-" * 100 )
        for function_name, execution_times in self.task_stats.items():
            print_line_stats(function_name.split("_")[0], execution_times)
        print("-"*100)
        print_line_stats("Mainloop", self.loop_stats)
        print("-"*100)
        print("Frequence:", round(self.loop_count / ( time.time()-self.loop_start ),3),"hz" )
        print("\n")