import csv
import torch
import torch.nn as nn
import torch.optim as optim

import yaml
config = None
with open(r'./config.yaml') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)


from src.dataset import build_dataset, load_dataset_to_gpu
from src.model import Fully_Connected
from src.device import device


input_size = 9 * 3 #can be player 1, player 2 or empty

net_description = [
    input_size,
		input_size*8,
    input_size*4,
    input_size*2,
    input_size,
    9
]

net = Fully_Connected(description=net_description).double().to(device)

loss_log = []

optimizer = optim.RMSprop(
    net.parameters(), lr=config["learning_rate"], momentum=0.9)
loss_func = nn.MSELoss()

# renseigner les données dans des fichiers excel
precision_file = open('precision.csv', 'w', newline='')
epochs_file = open('epochs.csv', 'w', newline='')
precision_writer = csv.writer(precision_file)
precision_writer.writerow(["step #", "Accuracy"])
epochs_writer = csv.writer(epochs_file)
epochs_writer.writerow(["epoch #", "Erreur Moyenne", "Précision"])

dataset = load_dataset_to_gpu(build_dataset())

good_answers = 0
answer_amount = 0
try:
	print("training model... ")
	for e in range(config['epochs']):
		r_loss = 0
		for i in range(len(dataset)):
			optimizer.zero_grad()
			in_vec, out_vec = dataset[i]

			net_out = net(in_vec)
			loss = loss_func(net_out, out_vec)
			r_loss += loss
			loss.backward()
			optimizer.step()

			#determine good answer
			l_v = out_vec.tolist()
			bestscore = max(l_v)
			if net_out.argmax() in [i for i, x in enumerate(l_v) if x == bestscore]:
				good_answers += 1
			answer_amount += 1
			#end of epoch
			percentage = 100*good_answers/(answer_amount)
			precision_writer.writerow([answer_amount, percentage])
			


		loss_log.append(r_loss)
		avrg_loss = r_loss/len(dataset)
		precision = 100*good_answers/answer_amount
		epochs_writer.writerow([e, avrg_loss, precision])
		print('Epoch: {} - Cumultative Loss: {:.6f} - Average Loss {:.6f}, - Accuracy {:.6f}'.format(e, r_loss, avrg_loss,precision))

except:
	pass

import matplotlib.pyplot as plt
torch.save(net.state_dict(), config['save_path'])
plt.plot(loss_log)
plt.show()
input()

epochs_file.close()
precision_file.close()


