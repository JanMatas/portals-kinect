from kalman import Kalman2D
class TrackedObject():
    def __init__(self, x, y):
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.started = True
        self.kalman = False

    def update (self, x, y):
        self.x = x
        self.y = y
        if self.started:
            print "start"
            v_x = x - self.start_x
            v_y = y - self.start_y
            print v_x, v_y
            self.kalman = Kalman2D(x, y, v_x, v_y)
            self.started = False
        else:
            print x, y
            self.kalman.predict_and_correct(x, y)

    def update_missing(self):
        if not self.started:
            self.kalman.predict()


    def get_position(self):
        return (self.x, self.y)

    def get_kalman(self):
        if self.kalman:
            return self.kalman.get_prediction()
        else:
            return self.get_position()
