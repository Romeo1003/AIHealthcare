# AIHealthcare
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'C:/Users/Uday Kiran/OneDrive/Desktop/VS CODE MAMA/Lung X-Ray Image/Lung X-Ray Image',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
<img width="779" alt="image" src="https://github.com/user-attachments/assets/ebc7cfc5-0088-442a-8fd0-55c22dedffd0">
validation_generator = train_datagen.flow_from_directory(
    'C:/Users/Uday /OneDrive/Desktop/VS CODE MAMA/Lung X-Ray Image/Lung X-Ray Image',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))


<img width="738" alt="image" src="https://github.com/user-attachments/assets/7a6a354e-885a-4487-9524-213912423fe8">

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=20
)

# Step 5: Plot Training History
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


<img width="630" alt="image" src="https://github.com/user-attachments/assets/36cae27a-634b-47e7-acc8-b1180750634f">
# Step 6: Save the Model
model.save('lung_disease_cnn_model.h5')
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, recall_score, f1_score
import seaborn as sns
# Step 7: Evaluate the Model
val_generator = validation_generator
val_generator.reset()
y_pred = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
<img width="911" alt="image" src="https://github.com/user-attachments/assets/82bd7a1c-2d57-4dec-814c-a54a26ba25cd">
# Metrics Calculation
accuracy = accuracy_score(y_true, y_pred_classes)
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')
auc = roc_auc_score(y_true, y_pred, multi_class='ovr’)

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

<img width="492" alt="image" src="https://github.com/user-attachments/assets/845fef86-4383-4f93-9eba-d911b563cb44">

