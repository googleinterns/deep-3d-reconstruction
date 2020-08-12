import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from datetime import datetime

IMAGE_HEIGHT = 968
IMAGE_WIDTH = 1296

class ConvNetAutoEncoder(tf.keras.Model):
    def __init__(self):
        super(ConvNetAutoEncoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.maxp1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.maxp2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')

        self.encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        
        self.conv4 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.upsample1 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv5 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.upsample2 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.upsample3 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv7 = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        
    def encoder(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        encoding = self.encoded(x)
        return encoding
    
    def decoder(self, x):
        x = self.conv4(x)
        x = self.upsample1(x)
        x = self.conv5(x)
        x = self.upsample2(x)
        x = self.conv6(x)
        x = self.upsample3(x)
        x = self.conv7(x)
        return x
        
    def call(self, x):
        encoding = self.encoder(x)
        reconstructed_data = self.decoder(encoding)
        return reconstructed_data
    

class ResNetAutoEncoder(tf.keras.Model):
    
    def __init__(self, filters=32, conv_size=2, channels=3, num_resnet_blocks=10):
        super(ResNetAutoEncoder, self).__init__()
        self.filters = filters
        self.conv_size = conv_size
        self.num_resnet_blocks = num_resnet_blocks
        
        # encoder
        self.conv1 = tf.keras.layers.Conv2D(filters, conv_size, activation='relu', padding='same')
        # self.bn1 = tf.keras.layers.BatchNormalization()
        self.max_pool = tf.keras.layers.MaxPooling2D(conv_size, padding='same')
        self.resnet_block = []
        for i in range(num_resnet_blocks):
            block = []
            block.append(tf.keras.layers.Conv2D(filters, conv_size, activation='relu', padding='same'))
            # block.append(tf.keras.layers.BatchNormalization())
            block.append(tf.keras.layers.Conv2D(filters, conv_size, activation=None, padding='same'))
            # block.append(tf.keras.layers.BatchNormalization())
            block.append(tf.keras.layers.Add())
            block.append(tf.keras.layers.Activation('relu'))
            self.resnet_block.append(block)    
        self.conv2 = tf.keras.layers.Conv2D(filters, conv_size, activation='relu', padding='same')
        self.encoded = tf.keras.layers.AveragePooling2D(conv_size, padding='same')
        
        # decoder
        self.conv3 = tf.keras.layers.Conv2D(filters, conv_size, activation='relu', padding='same')
        self.upsample1 = tf.keras.layers.UpSampling2D(conv_size)
        self.conv4 = tf.keras.layers.Conv2D(filters, conv_size, activation='relu', padding='same')
        self.upsample2 = tf.keras.layers.UpSampling2D(conv_size)    
        self.conv5 = tf.keras.layers.Conv2D(channels, conv_size, activation='sigmoid', padding='same')
    
    def encoder(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.max_pool(x)
        shortcut = x
        for i in range(self.num_resnet_blocks):
            for block in self.resnet_block[i]:
                try:
                    x = block(x)
                except:
                    x = block([x, shortcut])

        x = self.conv2(x)
        encoding = self.encoded(x)
        return encoding

    def decoder(self, x):
        x = self.conv3(x)
        x = self.upsample1(x)
        x = self.conv4(x)
        x = self.upsample2(x)
        x = self.conv5(x)
        return x

    def call(self, x):
        encoding = self.encoder(x)
        reconstructed_data = self.decoder(encoding)
        return reconstructed_data
    
    
def mse_loss(model, original):
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
    return reconstruction_error


def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)
        
    
def build_model(img_shape,lr, model_name='resnet'):
    if model_name == 'resnet':
        model = ResNetAutoEncoder()
    else:
        model = ConvNetAutoEncoder()
    model.build(img_shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), 
                  loss='mse',
                  metrics=['accuracy'])
    return model


def plot_reconstructed_images(target_images, predictions):
    n = target_images.shape[0]
    plt.figure(figsize=(15, 15))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(target_images[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(predictions[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    plt.show()


if __name__ == '__main__':
    parsed_data = input_fn(['tfdata/scene0004_00.tfrecords'])
    
    for data in parsed_data.take(1):
        train_images = data['images'][:12]
        test_images = data['images'][12:]
        
    logdir = "/logs"
    os.makedirs(logdir, exist_ok=True)
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=1, write_images=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    keras_callbacks = [tensorboard, early_stopping]
    
    image_shape = (None,IMAGE_HEIGHT,IMAGE_WIDTH,3)

    model = build_model(image_shape,lr=1e-3)
    model.fit(train_images, train_images,
                    epochs=100,
                    batch_size=4,
                    validation_data=(test_images, test_images),
                    callbacks=keras_callbacks)
    
    test_predictions = model.predict(test_images)
    plot_reconstructed_images(test_images, test_predictions)

    # writer = tf.summary.create_file_writer('tmp')
    # epochs = 50

    # autoencoder = ResNetAutoEncoder()
    # autoencoder.build(image_shape)
    # opt = tf.optimizers.Adam(learning_rate=1e-3)

    # with writer.as_default():
    #     with tf.summary.record_if(True):
    #         for epoch in range(epochs):
    #             for step, images in enumerate(train_images):
    #                 train(mse_loss, autoencoder, opt, images)
    #                 loss_values = mse_loss(autoencoder, images)
    #                 original = images
    #                 reconstructed = autoencoder(images)
    #                 tf.summary.scalar('loss', loss_values, step=step)
    #                 tf.summary.image('original', original, max_outputs=10, step=step)
    #                 tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)