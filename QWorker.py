from PySide6.QtCore import QThread, Signal
from queue import Queue

class Worker(QThread):
    finished = Signal()  # 모든 작업이 완료되었을 때 발생하는 시그널
    task_finished = Signal(str)  # 개별 작업이 완료되었을 때 발생하는 시그널
    error = Signal(str)  # 에러 발생 시 발생하는 시그널

    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks  # 실행할 작업 리스트
        self.queue = Queue()  # 작업 큐

        # 작업 큐에 작업 추가
        for task in tasks:
            self.queue.put(task)

    def run(self):
        while not self.queue.empty():
            task = self.queue.get()  # 큐에서 작업 꺼내기
            try:
                task()  # 작업 실행
                self.task_finished.emit(f"Task completed: {task.__name__}")
            except Exception as e:
                self.error.emit(f"Error in {task.__name__}: {str(e)}")
            finally:
                self.queue.task_done()  # 작업 완료 표시

        self.finished.emit()  # 모든 작업 완료 시그널