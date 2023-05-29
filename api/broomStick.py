from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import collections
import numpy as np
import pandas as pd



class broomStick:
    '''
    🧹: wrapper for cs maths
    '''
    def __init__(self, vector: List[float], name: str, y=None):
        if y is None:
            self.vector = np.array(vector)
        else:
            self.vector = np.column_stack((vector, y))
        
        self.name = name
        self.label = name
        self.parse_vector()
        
    def parse_vector(self):
        if self.vector.ndim == 1:
            self.dict = collections.Counter(self.vector)
            self.vals, self.cnt = zip(*self.dict.items())
            self.n = np.sum(self.cnt)
            self.prob_vector = np.array([x / self.n for x in self.cnt])
            self.mu = np.mean(self.vals)
            self.sigma = np.std(self.vals)
        elif self.vector.ndim == 2:
            self.mu = np.mean(self.vector, axis=0)
            self.sigma = np.std(self.vector, axis=0)
        else:
            raise ValueError("The input must be either 1D or 2D array.")

    def get_entropy(self) -> float:
        h = collections.Counter(self.vector)
        p = np.array([h[x] / self.n for x in h.keys()])
        self.p = p
        self.entropy = round(-np.sum(p * np.log2(p)), 2)
        print(self.entropy)
        return self.entropy

    def get_variance(self) -> float:
        self.variance = np.sum([(x - self.mu)**2 for x in self.vector]) / self.n
        return self.variance

    def get_norm_vector(self, x: Union[None, List[float]] = None) -> List[float]:
        if not x:
            x = self.vector
        v_min = np.min(x)
        v_max = np.max(x)
        v_max_min = v_max - v_min
        self.vector_norm = [(x-v_min)/(v_max_min) for x in x]
        return self.vector_norm

    def pdf(self) -> None:
        plt.plot(self.prob_vector())

    def get_ctl(self, samples: int = 1000) -> None:
        res = []
        n = len(self.vector)-1

        for dataPoint in range(samples):
            idxVector = [
                self.vector[np.random.randint(0, n)],
                self.vector[np.random.randint(0, n)],
                self.vector[np.random.randint(0, n)]
            ]
            rs = np.sum(idxVector) // len(idxVector)
            res.append(rs)
        plt.hist(res)
        plt.show()
        self.ctl_values = np.histogram(rs)

    def get_cdf(self, show: bool = True) -> None:
        values = np.array(self.prob_vector)
        cdf = values.cumsum() / values.sum()
        self.ccdf = 1-cdf
        self.cdf = cdf
        if show:
            plt.xscale("log")
            plt.yscale("log")
            plt.title(f"Cumulative Distribution")
            plt.ylabel("P(K) >= K")
            plt.xlabel("K")
            plt.plot(cdf[::-1])
            plt.show()

    def get_pdf_linear_binning(self) -> None:
        
        plt.xlabel("K")
        plt.ylabel("P(K)")
        plt.plot(self.vals, self.prob_vector, "o")
        plt.show()

    def get_pdf_log_binning(self, show: bool = True) -> None:

        in_max, in_min = max(self.prob_vector), min(self.prob_vector)
        log_bins = np.logspace(np.log10(in_min), np.log10(in_max))
        hist_cnt, log_bin_edges = np.histogram(self.prob_vector, bins=log_bins, density=True, range=(in_min, in_max))

        n = np.sum(hist_cnt)
        log_prob_vector = np.array([x / n for x in hist_cnt])

        if show:
            plt.title(f"Log Binning & Scaling")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("K")
            plt.ylabel("P(K)")
            plt.plot(log_bin_edges[:-1], log_prob_vector[::-1], "o")
            plt.show()

    def get_pdf_from_mu(self, idx: int = -1) -> None:
        c = collections.Counter(self.vector)
        if idx > -1:
            print("PDF_VALUE:", self.prob_vector[idx])

    def get_coverance(self, v2: List[float]) -> float:
        n = len(self.vector)
        y_mu = np.mean(v2)
        coverance = np.sum([(x - self.mu)*(y - y_mu) for x, y in zip(self.vector, v2)]) / n-1
        return coverance

    def get_slope(self, v2: List[float]) -> float:
        slope = self.get_coverance(v2) / self.get_variance()
        self.slope = slope
        return slope


    # Linear Regression
    def linear_regression(self, n2: 'Jiji') -> List[float]:
        x_mu = self.mu
        y_mu = n2.mu
        
        n = len(self.vector)
        top_term = sum((self.vector[i] - x_mu) * (n2.vector[i] - y_mu) for i in range(n))
        btm_term = sum((self.vector[i] - x_mu)**2 for i in range(n))

        m = top_term / btm_term
        b = y_mu - (m * x_mu)

        max_x = np.max(self.vector) + 10
        min_x = np.min(self.vector) - 10
        x_delta = np.linspace(min_x, max_x, 10)

        y_delta = b + m * x_delta

        plt.scatter(self.vector, n2.vector)
        plt.plot(x_delta, y_delta, 'ro')
        plt.show()
        return y_delta

    def lr_alg(self, x: List[float], y: Union[List[float], None] = []) -> Tuple[float, float]:
        if len(y) < 1:
            y = x
            x = [timeStep for timeStep in range(len(y))]

        ones = np.ones(len(x))
        X = np.column_stack((ones, x))

        Xt_X_inv = np.linalg.inv(np.dot(X.T, X))
        theta = np.dot(np.dot(Xt_X_inv, X.T), y)

        intercept, slope = theta[0], theta[1]

        y_pred = np.dot(X, theta)

        plt.plot(x, y, label='Actual', linestyle=':', color='blue', marker='^')
        plt.plot(x, y_pred, color='red', label='Predicted', linestyle='-')
        plt.title('Curve Line Prediction')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
        return intercept, slope
    

    # KNN
    def simple_knn(self, target: bool = False, knnSize: int = 5) -> None:
        self.knn_init()
        self.distanceVector = collections.defaultdict()
        cord_vector = list(self.graph.values())
        x_y = [(p.x, p.y) for p in self.nodeList]
        x, y = zip(*x_y)
        fig, ax = plt.subplots()

        ax.scatter(x, y, s=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.show()

    def knn_init(self) -> None:
        self.create_node_list()
        self.create_graph()

    def create_node_list(self) -> None:
        nodeList = []
        adjList = collections.defaultdict()

        for idx in range(len(self.vector) - 1):
            dx = self.vector[idx]
            dy = self.p2.vector[idx]
            delta = Node(   dx, 
                            dy,
                            idx,
                            self.label         
                        )
            nodeList.append(delta)
            adjList[(dx,dy)] = delta

        
        self.nodeList = nodeList
        self.adjList = adjList
        print('nodelist', self.nodeList)
    
    def create_graph(self) -> None:
        graph = collections.defaultdict(list)

        for node in self.nodeList:
            x, y = node.x, node.y
            graph[(x, y)].append(node)
            

        delta = list(graph.keys())
        x, y = zip(*delta)
        self.scatterGraph = (x, y)
        self.graph = graph

    # NPL
    def freq_count(self, path: str) -> Tuple[List[Tuple[str, float]], float, List[Tuple[str, int]]]:
        with open(path, 'r') as file:
            book = file.read()

        char_freq = collections.Counter(book.lower())
        filtered_keys = sorted(
            [(key, val) for key, val in char_freq.items() if key.isalpha()],
            key=lambda x: x[1]
        )

        x, y = zip(*filtered_keys)

        n = np.sum(y)
        prob_vector = list(zip(x, [cnt / n for cnt in y]))
        px, py = zip(*prob_vector)

        mu = np.mean(y)

        plt.xlabel('Char')
        plt.ylabel('Frequency')
        plt.title('Frequency of Char')
        plt.scatter(x, y)
        plt.show()

        plt.xlabel('K')
        plt.ylabel('P(K)')
        plt.title('Prob Vector')
        plt.scatter(px, py)
        plt.show()

        return prob_vector, mu, filtered_keys

class Node():
    
    def __init__(self,x,y,idx,label) -> None:
        self.x = x 
        self.y = y 
        self.idx = idx
        self.label = label

def TestbroomStick():
    print('start test')
    # Linear Regression 
    x = sorted([1,1,1,100,5,6,7,80,90,100])
    y = sorted([2,2,2,50,5,6,7,40,45,50])
    jiji1 = broomStick(x, 'jiji1')
    jiji2 = broomStick(y, 'jiji2')
    jiji1.parse_vector()
    jiji2.parse_vector()
    print('entropy')
    jiji1.get_entropy()
    jiji1.get_variance()
    jiji1.get_norm_vector()
    jiji1.get_pdf_linear_binning()
    jiji1.get_pdf_log_binning()

    jiji1.get_pdf_from_mu()
    jiji1.get_coverance(jiji2.vector)
    jiji1.get_slope(jiji2.vector)
    jiji1.linear_regression(jiji2)
    jiji1.lr_alg(x, y)
    jiji1.p2 = jiji2
    jiji1.simple_knn()
    #jiji1.freq_count('path/to/file.txt')



if __name__ == '__main__':
    TestbroomStick()







        




