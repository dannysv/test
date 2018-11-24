import gensim
import codecs


def format_lines(attib_file):
    with open(attib_file) as f:
        lines = f.readlines()
        arr_letras = []
        arr_pesos = []
        for i in range(len(lines)):
            if(i%2==0):
                arr_letras.append(lines[i].strip().split(','))
            else:                                                                               arr_pesos.append(lines[i].strip().split(','))
    return (arr_letras, arr_pesos)


####################### Evaluation ####################
from sklearn.metrics import classification_report

def evaluation(true, predict, domain):
    true_label = []
    predict_label = []

    if domain == 'restaurant':
        for line in predict:
            predict_label.append(line.strip())
        for line in true:
            true_label.append(line.strip())
    print(classification_report(true_label, predict_label, ['Food', 'Staff', 'Ambience', 'Anecdotes', 'Price', 'Miscellaneous'], digits=3))
############## Open attention files #################
domain = 'restaurant'
print('--- Results on %s domain ----' %(domain))
#test_labels = '/home/danny/gpuimp/experiments/v3_mymodel_onebyone/modified_abae_v3_mymodel_food/preprocessed_data/%s/test_label.txt' % (domain)

test_labels = '/home/danny/gpuimp/experiments/modified_abae/preprocessed_data/%s/test_label.txt' % (domain)

evaluation(open(test_labels), open('test_labels_abae.txt'), domain)

