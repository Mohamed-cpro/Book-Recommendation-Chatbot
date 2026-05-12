# Generated from: RNN Model 3 — BiLSTM with Attention.ipynb
# Converted at: 2026-05-12T23:25:01.653Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# RNN Model 3 — BiLSTM with Attention


def build_bilstm_attention(vocab=10_000, maxlen=64, num_classes=NUM_CLASSES):
    inp = layers.Input(shape=(maxlen,))
    x   = layers.Embedding(vocab, 32, mask_zero=False)(inp)
    x   = layers.SpatialDropout1D(0.5)(x)
    x   = layers.Bidirectional(layers.LSTM(32, return_sequences=True,
                                           dropout=0.4, recurrent_dropout=0.3,
                                           kernel_regularizer=L2))(x)
    x   = layers.Bidirectional(layers.LSTM(16, return_sequences=True,
                                           dropout=0.4, recurrent_dropout=0.3))(x)
    x   = AttentionLayer()(x)
    x   = layers.Dense(32, activation='relu', kernel_regularizer=L2)(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    m   = models.Model(inp, out, name='RNN_BiLSTM_Attention')
    m.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m
rnn3 = build_bilstm_attention()
rnn3.summary()

# Training BiLSTM Attention


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



print('\nTraining BiLSTM Attention')
h_rnn3 = train_rnn(rnn3, X_tr_seq, y_tr, X_val_seq, y_val)
rnn3.save(os.path.join(OUTPUT_DIR, 'rnn3_bilstm_attention.keras'))

from sklearn.metrics import classification_report
print('Final TEST set evaluation:')
for name, model in [ ('BiLSTM+Attention', rnn3)]:
    loss, acc = model.evaluate(X_te_seq, np.array(y_te), verbose=0)
    print(f'  {name:22s}  test_acc={acc:.4f}  test_loss={loss:.4f}')

for name, model in [ ('BiLSTM+Attention', rnn3)]:
    preds = np.argmax(model.predict(X_te_seq, verbose=0), axis=1)
    print(f'\n{name}:')
    print(classification_report(y_te, preds, target_names=le.classes_))