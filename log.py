import matplotlib.pyplot as plt

type = "generator" # retriever or generator

# Open log file and read each line
model = "bert-base-augmented-2"
log_file = "output\\"+model+"-"+type+".txt"
batch_size = None
report_interval = None

# train_data_size = 6251
train_data_size = 27718

total_time = 0
validation_accuracy_3 = []
validation_accuracy_5 = []

iterations = []
loss = []

exe_accuracy = []
prog_accuracy = []

with open(log_file, 'r') as f:
  for line in f:
    line = line.strip()
    # If the line contains the time, extract it
    if "time" in line:
      time = line.split(" ")[-1]
      total_time += float(time)

    # OBS space is import to avoid matching "batch_size_test" in "report_test"
    if "batch_size " in line:
      batch_size = int(line.split(" ")[-1])
      print("Batch size:", batch_size)
    
    if "report " in line:
      report_interval = int(line.split(" ")[-1])

    ## If the line contains the validation accuracy, extract it
    elif "Top 3" in line:
      accuracy = line.split(" ")[-1]
      validation_accuracy_3.append(float(accuracy))
    elif "Top 5" in line:
      accuracy = line.split(" ")[-1]
      validation_accuracy_5.append(float(accuracy))

    elif "exe acc:" in line:
      _exe_accuracy = line.split(" ")[2]
      _prog_accuracy = line.split(" ")[-1]
      exe_accuracy.append(float(_exe_accuracy))
      prog_accuracy.append(float(_prog_accuracy))

    # If the line contains the loss, extract it
    elif ": loss" in line:
      parts = line.split(" ")
      loss.append(float(parts[-1]))
      iterations.append(int(parts[0]))
    


print("Total time: ", round(total_time, 2), " seconds")
## Time in hours
print("Total time: ", round(total_time/60/60, 2), " hours")

print("Average time per iteration: ", round(total_time/len(iterations), 2), " seconds")
epochs = round((batch_size*iterations[-1])/train_data_size,1)
print("Epochs: ", epochs)


# ## Retriever
# ## Max validation accuracy index
if type == "retriever":
  max_index_3 = validation_accuracy_3.index(max(validation_accuracy_3))
  print("Best top 3 validation",  max_index_3 + 1, ":", validation_accuracy_3[max_index_3])

  max_index_5 = validation_accuracy_5.index(max(validation_accuracy_5))
  print("Best top 5 validation",  max_index_5 + 1, ":", validation_accuracy_5[max_index_5])




print()
# Last validation accuracy index
if type == "retriever":
  print("Last top 3 validation",  len(validation_accuracy_3), ":", validation_accuracy_3[-1])
  print("Last top 5 validation",  len(validation_accuracy_5), ":", validation_accuracy_5[-1])

# Generator
# Max validation accuracy index
if type == "generator":
  max_index_exe = exe_accuracy.index(max(exe_accuracy))
  print("Best exe accuracy",  max_index_exe + 1, ":", exe_accuracy[max_index_exe])

  max_index_prog = prog_accuracy.index(max(prog_accuracy))
  print("Best prog accuracy",  max_index_prog + 1, ":", prog_accuracy[max_index_prog])





# Create a figure with one row and two columns of subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

# Plot the loss in the first subplot
ax1.plot(iterations, loss)
ax1.set_title("%s %s - loss (%s batch size) %s epochs" % (type, model, batch_size, epochs))
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Loss")


# ## Retriever
# ## Plot the validation accuracy in the second subplot
if type == "retriever":
  ax2.plot([x * report_interval for x in range(1, len(validation_accuracy_3) + 1)], validation_accuracy_3, label="Top 3")
  ax2.plot([x * report_interval for x in range(1, len(validation_accuracy_5) + 1)], validation_accuracy_5, label="Top 5")
  ax2.set_title("Retriever %s - validation accuracy (%s batch size) %s epochs" % (model, batch_size, epochs))
  ax2.set_xlabel("Iterations")
  ax2.set_ylabel("Validation accuracy")
  ax2.legend()


## Generator
## Plot the exe accuracy and prog accuracy in the second subplot
if type == "generator":
  ax2.plot([x * report_interval for x in range(1, len(exe_accuracy) + 1)], exe_accuracy, label="Exe")
  ax2.plot([x * report_interval for x in range(1, len(exe_accuracy) + 1)], prog_accuracy, label="Prog")
  ax2.set_title("%s %s - accuracy (%s batch size) %s epochs" % (type, model, batch_size, epochs))
  ax2.set_xlabel("Iterations")
  ax2.set_ylabel("Accuracy")
  ax2.legend()

# Show the plot
plt.show()

