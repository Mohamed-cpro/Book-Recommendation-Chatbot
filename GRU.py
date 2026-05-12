# Generated from: GRU.ipynb
# Converted at: 2026-05-12T23:23:30.226Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# RNN Model 2 — Bidirectional GRU


def build_bigru(vocab=10_000, maxlen=64, num_classes=NUM_CLASSES):
    inp = layers.Input(shape=(maxlen,))
    x   = layers.Embedding(vocab, 32, mask_zero=True)(inp)
    x   = layers.SpatialDropout1D(0.5)(x)
    x   = layers.Bidirectional(layers.GRU(32, return_sequences=True,
                                          dropout=0.4, recurrent_dropout=0.3,
                                          kernel_regularizer=L2))(x)
    x   = layers.Bidirectional(layers.GRU(16, dropout=0.4,
                                          recurrent_dropout=0.3))(x)
    x   = layers.Dense(32, activation='relu', kernel_regularizer=L2)(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    m   = models.Model(inp, out, name='RNN_BiGRU')
    m.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

rnn2 = build_bigru()

#  Train RNN bigru 


class AccuracyCap(tf.keras.callbacks.Callback):
    """Stops training when val_accuracy exceeds the cap."""
    def __init__(self, cap=0.85):
        super().__init__()
        self.cap = cap

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_accuracy', 0)
        if val_acc >= self.cap:
            print(f'\nVal accuracy reached {val_acc:.4f} — stopping at cap {self.cap}.')
            self.model.stop_training = True


def train_rnn(model, Xtr, ytr, Xval, yval, epochs=7):
    cb = [
        AccuracyCap(cap=0.85),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_loss', min_lr=1e-6),
        callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, f'{model.name}_best.keras'),
            save_best_only=True, monitor='val_accuracy'
        )
    ]
    return model.fit(
        Xtr, np.array(ytr),
        validation_data=(Xval, np.array(yval)),
        epochs=epochs, batch_size=32, callbacks=cb
    )


print('\nTraining BiGRU')
h_rnn2 = train_rnn(rnn2, X_tr_seq, y_tr, X_val_seq, y_val)
rnn2.save(os.path.join(OUTPUT_DIR, 'rnn2_bigru.keras'))

from sklearn.metrics import classification_report
print('Final TEST set evaluation:')
for name, model in [ ('BiGRU', rnn2)]:
    loss, acc = model.evaluate(X_te_seq, np.array(y_te), verbose=0)
    print(f'  {name:22s}  test_acc={acc:.4f}  test_loss={loss:.4f}')

for name, model in [('BiLSTM', rnn1), ('BiGRU', rnn2), ('BiLSTM+Attention', rnn3)]:
    preds = np.argmax(model.predict(X_te_seq, verbose=0), axis=1)
    print(f'\n{name}:')
    print(classification_report(y_te, preds, target_names=le.classes_))