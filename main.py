import pandas as pd
from pandas_profiling import ProfileReport
import os.path
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
from dtreeviz.trees import *


def load_data():
    df = pd.read_csv('input/Recruiting_Task_InputData.csv')

    if os.path.isfile('output/Recruiting_Task_InputData_PP_report.html'):
        print("File exists")
    else:
        profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        profile.to_file('output/Recruiting_Task_InputData_PP_report.html')
    return df


def prepro(df):
    raw_csv_data = df.dropna().values
    # split into input (X) and output (y) variables
    X = raw_csv_data[:, 1:-1]
    y = raw_csv_data[:, -1]

    ohe = OrdinalEncoder()
    ohe.fit(X)
    X_enc = ohe.transform(X)

    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)

    # Standardize
    scaled_inputs = preprocessing.scale(X_enc)

    # Shuffle
    shuffled_indices = np.arange(scaled_inputs.shape[0])
    np.random.shuffle(shuffled_indices)

    shuffled_inputs = scaled_inputs[shuffled_indices]
    shuffled_targets = y_enc[shuffled_indices]

    # Split 80-10-10
    samples_count = shuffled_inputs.shape[0]
    train_samples_count = int(0.8*samples_count)
    validation_samples_count = int(0.1 * samples_count)
    test_sample_count = samples_count - train_samples_count - validation_samples_count

    train_inputs = shuffled_inputs[:train_samples_count]
    train_targets = shuffled_targets[:train_samples_count]
    validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
    validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]
    test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
    test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

    print(np.sum(train_targets), train_samples_count, np.sum(train_targets)/train_samples_count)

    np.savez('output\data_train',
             inputs=train_inputs, targets=train_targets)
    np.savez('output\data_validation',
             inputs=validation_inputs, targets=validation_targets)
    np.savez('output\data_test',
             inputs=test_inputs, targets=test_targets)

    return train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets


def task_dl():
    npz = np.load('output\data_train.npz')
    train_inputs = npz['inputs'].astype(np.float)
    train_targets = npz['targets'].astype(np.int)
    npz = np.load('output\data_validation.npz')
    validation_inputs = npz['inputs'].astype(np.float)
    validation_targets = npz['targets'].astype(np.int)
    npz = np.load('output\data_test.npz')
    test_inputs = npz['inputs'].astype(np.float)
    test_targets = npz['targets'].astype(np.int)

    # Model generation
    input_size = 8
    output_size = 2
    hidden_layer_size = 48
    NUM_EPOCHS = 350
    BATCH_SIZE = 20
    learn_rate = 0.0005
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=4)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_size),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
        tf.keras.layers.Dense(output_size, activation='softmax')
    ])

    m_optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate, epsilon=0.1)
    m_losses = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=m_optimizer, loss=m_losses, metrics=['accuracy'])
    history = model.fit(train_inputs, train_targets,
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[early_stopping],
                        validation_data=(validation_inputs, validation_targets),
                        verbose=2)

    # Get training and test loss histories
    training_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(epoch_count, training_loss, 'r--')
    ax1.plot(epoch_count, val_loss, 'b-')
    ax1.legend(['Training Loss', 'Test Loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2.plot(epoch_count, train_acc, 'r--')
    ax2.plot(epoch_count, val_acc, 'b-')
    ax2.legend(['Training Acc', 'Test Acc'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    plt.savefig('output/loss_acc_ips_'+str(input_size)+'_ops_'+str(output_size)+'_hls_'+
                str(hidden_layer_size)+'_bs_'+str(BATCH_SIZE)+'_lr_'+str(learn_rate)+'.png', bbox_inches = "tight")
    plt.close(fig)
    test_loss, test_acc = model.evaluate(test_inputs, test_targets)
    print(test_loss, test_acc * 100)

    class_names = ['response', 'no response']
    feature_names = ['age', 'lifestyle', 'zip code', 'family status', 'car', 'sports', 'earnings', 'living area']

    # Sample the training set to accelerate analysis
    df_train_normed_summary = shap.utils.sample(train_inputs, 250)

    # Instantiate an explainer with the model predictions and training data summary
    explainer = shap.KernelExplainer(model.predict, df_train_normed_summary)

    # Extract Shapley values from the explainer
    shap_values = explainer.shap_values(df_train_normed_summary)

    values = shap_values[0]
    base_values = [explainer.expected_value[0]] * len(shap_values[0])

    tmp = shap.Explanation(values=np.array(values, dtype=np.float32),
                           base_values=np.array(base_values, dtype=np.float32),
                           data=np.array(df_train_normed_summary),
                           feature_names=feature_names)

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches

    fig = shap.plots.waterfall(tmp[0], show=False)
    plt.savefig('output/shap_waterfall_ips_'+str(input_size)+'_ops_'+str(output_size)+'_hls_'+
                str(hidden_layer_size)+'_bs_'+str(BATCH_SIZE)+'_lr_'+str(learn_rate)+'.png', bbox_inches = "tight")
    plt.close(fig)

    fig = shap.plots.beeswarm(tmp, show=False)
    plt.savefig('output/shap_sum_dot_ips_'+str(input_size)+'_ops_'+str(output_size)+'_hls_'+
                str(hidden_layer_size)+'_bs_'+str(BATCH_SIZE)+'_lr_'+str(learn_rate)+'.png', bbox_inches = "tight")
    plt.close(fig)

    fig = shap.summary_plot(tmp, df_train_normed_summary, show=False, plot_type="bar")
    plt.savefig('output/shap_sum_bar_ips_'+str(input_size)+'_ops_'+str(output_size)+'_hls_'+
                str(hidden_layer_size)+'_bs_'+str(BATCH_SIZE)+'_lr_'+str(learn_rate)+'.png', bbox_inches = "tight")
    plt.close(fig)

    for i in range(df_train_normed_summary.shape[1]):
        feature = str('Feature ' + str(i))
        print(feature)
        fig = shap.dependence_plot(feature, shap_values[0], df_train_normed_summary, show=False)
        plt.savefig('output/'+str(feature_names[i])+'.png', bbox_inches = "tight")
        plt.close(fig)


def task_ml(train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets):
    class_names = ['no response','response']
    feature_names = ['age', 'lifestyle', 'zip code', 'family status', 'car', 'sports', 'earnings', 'living area']
    print('Dtree Start: ', datetime.datetime.now())

    clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(train_inputs, train_targets)

    y_pred = clf.predict(validation_inputs)
    report = classification_report(validation_targets, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv('output/report.csv', sep=';')

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, test_inputs, test_targets,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.savefig('output/cm.png', format="png", bbox_inches = "tight")

    viz = dtreeviz(clf,
                   x_data=test_inputs,
                   y_data=test_targets,
                   target_name='Klassen',
                   feature_names=feature_names,
                   class_names=class_names,
                   title="Decision Tree")
    viz.save('output/dtree.svg')


if __name__ == '__main__':
    df = load_data()
    train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = prepro(df)
    task_dl()
    task_ml(train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets)


