import matplotlib.pyplot as plt

# Open log file and read each line
log_file = "output\log-roberta-large-retreiever.txt"
batch_size = 16
train_data_size = 6251

total_time = 0
validation_accuracy_3 = []
validation_accuracy_5 = []

iterations = []
loss = []

with open(log_file, 'r') as f:
  for line in f:
    line = line.strip()
    # If the line contains the time, extract it
    if "time" in line:
      time = line.split(" ")[-1]
      total_time += float(time)
    
    # If the line contains the validation accuracy, extract it
    elif "Top 3" in line:
      accuracy = line.split(" ")[-1]
      validation_accuracy_3.append(float(accuracy))
    elif "Top 5" in line:
      accuracy = line.split(" ")[-1]
      validation_accuracy_5.append(float(accuracy))

    # If the line contains the loss, extract it
    elif "loss" in line:
      parts = line.split(" ")
      loss.append(float(parts[-1]))
      iterations.append(int(parts[0]))
    


print("Total time: ", round(total_time, 2), " seconds")
## Time in hours
print("Total time: ", round(total_time/60/60, 2), " hours")

print("Average time per iteration: ", round(total_time/len(iterations), 2), " seconds")
epochs = round((batch_size*iterations[-1])/train_data_size,1)
print("Epochs: ", epochs)

# Max validation accuracy index
max_index_3 = validation_accuracy_3.index(max(validation_accuracy_3))
print("Best top 3 validation",  max_index_3 + 1, ":", validation_accuracy_3[max_index_3])

max_index_5 = validation_accuracy_5.index(max(validation_accuracy_5))
print("Best top 5 validation",  max_index_5 + 1, ":", validation_accuracy_5[max_index_5])

# Plot the loss
plt.plot(iterations, loss)
plt.title("Retriever RoBERTa-large - loss (%s batch size) %s epochs" % (batch_size, epochs))
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
