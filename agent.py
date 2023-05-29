from typing import List
from api.lantern import lantern
from api.broomStick import broomStick
from api.crystalBall import crystalBall
import random

class Jiji(broomStick, lantern, crystalBall):
    '''
    🏮: wrapper for visuals of a neural net

    🧹: utils class for vector maths

    🔮: npl via dpr 
    '''
    
    def __init__(self, x: List[float], name: str, y=[]):
        broomStick.__init__(self, x, name, y)
        lantern.__init__(self, len(x), 4, len(y[0]))
        crystalBall.__init__(self, 'markdown example')
        #crystalBall.init_embeddings(file_path='/')
        print("init complete")



if __name__ == "__main__":
    # x = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    # y = [[0.4, 0.8], [0.6, 0.4], [0.8, 0.2]]
   
    #print(Kiki.get_norm_vector(x))
    #Kiki.testNet(x,y)
    # Kiki = Jiji(None,"test",None)
    agent = crystalBall('RD EX One')
    prompt_path = '/Users/kjams/Desktop/Jiji/api/trainingData/prompts.txt'
    ans_path = '/Users/kjams/Desktop/Jiji/api/trainingData/answers.txt'
    #x = agent.init_embeddings(prompt_path)
    #y = agent.init_embeddings(ans_path)
    agent.init_embeddings([prompt_path, ans_path])
    
    print(agent.embeddings.items())
    # x = agent.init_embeddings([prompt_path, ans_path])

    # for key, value in x.items():
    #     print(key,':',value[:5])
    #     print('')


