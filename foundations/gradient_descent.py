class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        x_new = init
        for _ in range(iterations):
            x_new = x_new - learning_rate*2*x_new
        return round(x_new,5)
