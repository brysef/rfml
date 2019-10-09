from rfml.data import build_dataset
from rfml.nn.eval import (
    compute_accuracy,
    compute_accuracy_on_cross_sections,
    compute_confusion,
)
from rfml.nn.model import build_model
from rfml.nn.train import build_trainer, PrintingTrainingListener

train, val, test, le = build_dataset(dataset_name="RML2016.10a")
model = build_model(model_name="CNN", input_samples=128, n_classes=len(le))
trainer = build_trainer(
    strategy="standard", max_epochs=3, gpu=True
)  # Note: Disable the GPU here if you do not have one
trainer.register_listener(PrintingTrainingListener())
trainer(model=model, training=train, validation=val, le=le)
acc = compute_accuracy(model=model, data=test, le=le)
acc_vs_snr, snr = compute_accuracy_on_cross_sections(
    model=model, data=test, le=le, column="SNR"
)
cmn = compute_confusion(model=model, data=test, le=le)

# Calls to a plotting function could be inserted here
# For simplicity, this script only prints the contents as an example
print("===============================")
print("Overall Testing Accuracy: {:.4f}".format(acc))
print("SNR (dB)\tAccuracy (%)")
print("===============================")
for acc, snr in zip(acc_vs_snr, snr):
    print("{snr:d}\t{acc:0.1f}".format(snr=snr, acc=acc * 100))
print("===============================")
print("Confusion Matrix:")
print(cmn)

model.save("cnn.pt")
