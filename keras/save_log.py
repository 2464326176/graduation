train_log = model.fit_generator(
train_generator,
steps_per_epoch = nb_train_samples        // batch_size,
epochs = epochs,
validation_data = validation_generator,
validation_steps  =nb_validation_samples  // batch_size,
)


# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), train_log.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), train_log.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), train_log.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), train_log.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on sar classifier")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("Loss_Accuracy_alexnet_{:d}e.jpg".format(epochs))
