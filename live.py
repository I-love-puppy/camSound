import tensorflow as tf

# SavedModel 형식으로 로드된 모델
model = tf.keras.models.load_model("sound_classifier_model")

# Keras H5 형식으로 저장
model.save("C:/Users/USER/Project/IoT/sound/sound_classifier_model.h5", save_format='h5')
