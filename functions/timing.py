import time


class Timer:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.time_elapsed = 0

    def start(self):
        self.start_time = time.time()

    def stop(self, text="Loading"):
        self.end_time = time.time()
        self.time_elapsed = self.end_time - self.start_time
        if self.time_elapsed < 150:
            print('Timing:: took %d second(s) %s'
                  '\n-----------------------------------' % (self.time_elapsed, text))
            self.start_time, self.end_time = 0, 0
        else:
            time_in_minutes = self.time_elapsed / 60
            print('Timing:: took %d minutes %s'
                  '\n-----------------------------------' % (time_in_minutes, text))
            self.start_time, self.end_time = 0, 0
