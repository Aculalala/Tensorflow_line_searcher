import tensorflow as tf

import numpy as np

class TF_Line_Searcher():
    def __init__(self, session, loss, optimizer=tf.train.GradientDescentOptimizer(1),Xterminate=1e-7,max_step=1000,multiplier=np.sqrt(10),
                 feed={}, gradient_alter=lambda x:x):
        self.session = session
        self.optimizer = optimizer
        self.multiplier = multiplier
        self.Xterminate = Xterminate
        self.max_step = max_step
        self.feed = feed
        self.loss = loss
        self.step_size = 0.01
        self.gradient_alter = gradient_alter
        self.memory = {}  # Backup weights
        self.Gradients = {}  # Obtained by taking diff

    def reset(self):
        self.step_size = 0.01
        self.memory = {}

    def __diff__(self):
        self.session.run(self.optimizer.minimize(self.loss), feed_dict=self.feed)
        for var in tf.trainable_variables():
            self.Gradients[var.name] = self.session.run(var) - self.memory[var.name]
            self.Gradients[var.name]=self.gradient_alter(self.Gradients[var.name])
        self.__restore__()

    def __backup__(self):
        #self.memory = {}
        for var in tf.trainable_variables():
            self.memory[var.name] = self.session.run(var)


    def __restore__(self):
        for var in tf.trainable_variables():
            self.session.run(tf.assign(var, self.memory[var.name]))

    def __apply__(self, ss):
        for var in tf.trainable_variables():
            self.session.run(tf.assign(var, self.memory[var.name] + self.Gradients[var.name] * ss))

    def __estimate__(self, ss):
        # self.__restore__()
        self.__apply__(ss)
        estimated_loss = self.session.run(self.loss, feed_dict=self.feed)
        self.__restore__()
        return estimated_loss

    def one_step(self):
        start_loss = self.session.run(self.loss, feed_dict=self.feed)
        last_loss = start_loss

        self.__backup__()
        self.__diff__()

        best_step_size = 0
        start_step_size = self.step_size
        while True:
            new_loss = self.__estimate__(self.step_size)
            if new_loss < last_loss:
                best_step_size = self.step_size
                self.step_size *= self.multiplier
                last_loss = new_loss
            else:
                break

        if best_step_size != 0:
            self.__apply__(best_step_size)
            assert (last_loss < start_loss)
            return True

        self.step_size=start_step_size
        while self.step_size > self.Xterminate:
            self.step_size /= self.multiplier
            new_loss = self.__estimate__(self.step_size)
            if new_loss < start_loss:
                self.__apply__(self.step_size)
                assert(new_loss<start_loss)
                return True
        self.reset()
        return False

    def auto(self):
        for i in range(self.max_step):
            if not self.one_step():
             return False #Terminated by step size
        return True #Terminated by step limit







