import pickle
import matplotlib.pyplot as plt

'''
function for generating loss graphs
'''

def loss_graph():
    '''show loss graph'''
    with open("dq_agents/doors_agents/doors_agents_loss/dq_doors_loss_values_10.pkl", "rb") as f:
        loss_values = pickle.load(f)

    interval = 50
    epochs = [i * interval for i in range(len(loss_values))]
        
    # plot loss values
    plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b', label='Training Loss')
    # labels
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss per Episode")
    # show graph
    plt.show()
    
loss_graph()