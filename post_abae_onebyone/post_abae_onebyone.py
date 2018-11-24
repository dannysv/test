import gensim
import codecs

#load model
#model = gensim.models.KeyedVectors.load_word2vec_format('../../Downloads/word2vec-master/sentences_yelp_text_pp_200.txt', binary=False)

model = gensim.models.KeyedVectors.load_word2vec_format('../post_abae_original/model_200_abae.txt', binary=False)

def format_lines(attib_file):
    with open(attib_file) as f:
        lines = f.readlines()
    arr_letras = []
    arr_pesos = []
    for i in range(len(lines)):
        if(i%2==0):
            arr_letras.append(lines[i].strip().split(','))
        else:
            arr_pesos.append(lines[i].strip().split(','))
    return (arr_letras, arr_pesos)

def mean_simil_contex(word, peso, peso_ref, refword, negatives):
    res = model.most_similar(positive=[refword, word], negative=negatives)
    val_t =0
    for i in range(5):
        simil = 0 
        try:
            simil = model.similarity(refword, res[i][0])
        except Exception as e:
            print(e)
        val = float(peso)*simil +  float(peso_ref)*simil 
        val_t+=val 
    return val_t

def mean_mean_sim_context(ar_sentence, pesos, pesos_ref,refword, negatives):
    total = 0
    for pword, peso, peso_ref in zip(ar_sentence, pesos, pesos_ref):
        try:
            if(pword!='<unk>'):
                total += mean_simil_contex(pword, peso, peso_ref,refword, negatives)
        except Exception as e:
            print(e)
    return total 

def similarity_val(word, letras, pesos):
    existen = 0
    total = 0
    for (letra, peso) in zip(letras, pesos):
        try:
            simil = model.wv.similarity(w1=word, w2=letra)
            simil = float(peso)*simil
            existen+=1
        except Exception as e:                                                              
            simil = 0
        total+=simil
    if total == 0:
        return 0
    else:
        return total/existen

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

def mayor3(x,y,z):
    mayor = -1000
    index = -1
    if(x>mayor):
        mayor=x
        index = 0
    if(y>mayor):
        mayor=y
        index = 1
    if(z>mayor):
        mayor=z
        index = 2
    return (mayor, index)

############## Open attention files #################
f_attrib_weights_staff = '/home/danny/gpuimp/experiments/v3_mymodel_onebyone/modified_abae_v3_mymodel_staff/code/output_dir_mymodel/restaurant/att_weights'
f_attrib_weights_ambience = '/home/danny/gpuimp/experiments/v3_mymodel_onebyone/modified_abae_v3_mymodel_ambience/code/output_dir_mymodel/restaurant/att_weights'
f_attrib_weights_food = '/home/danny/gpuimp/experiments/v3_mymodel_onebyone/modified_abae_v3_mymodel_food/code/output_dir_mymodel/restaurant/att_weights'

f_attrib_weights_rf1 = '/home/danny/gpuimp/experiments/v3_mymodel_onebyone/modified_abae_v3_mymodel_staff/code/output_dir_mymodel/restaurant/att_weights_rf'
f_attrib_weights_rf2 = '/home/danny/gpuimp/experiments/v3_mymodel_onebyone/modified_abae_v3_mymodel_ambience/code/output_dir_ambience/restaurant/att_weights_rf'
f_attrib_weights_rf3 = '/home/danny/gpuimp/experiments/v3_mymodel_onebyone/modified_abae_v3_mymodel_food/code/output_dir_food/restaurant/att_weights_rf'

(ar_letras_staff,ar_pesos_staff)= format_lines(f_attrib_weights_staff)
(ar_letras_ambience,ar_pesos_ambience)= format_lines(f_attrib_weights_ambience)
(ar_letras_food,ar_pesos_food)= format_lines(f_attrib_weights_food)
(ar_letras_rf1, ar_pesos_rf1) = format_lines(f_attrib_weights_rf1)
(ar_letras_rf2, ar_pesos_rf2) = format_lines(f_attrib_weights_rf2)
(ar_letras_rf3, ar_pesos_rf3) = format_lines(f_attrib_weights_rf3)


print(len(ar_letras_staff))
print(len(ar_pesos_staff))
print(len(ar_letras_rf1))
print(len(ar_pesos_rf1))
print(len(ar_letras_rf2))
print(len(ar_pesos_rf2))
print(len(ar_letras_rf3))
print(len(ar_pesos_rf3))


def method1():
    refwords = ['staff', 'ambience', 'food']
    result = []
    for refword in refwords:
        out = codecs.open(refword+'_simils.txt', 'w', 'utf-8')
        refword_vals = []
        for (letras_staff, pesos_staff, letras_ambience, pesos_ambience, letras_food, pesos_food, letras_rf1, pesos_rf1, letras_rf2, pesos_rf2, letras_rf3, pesos_rf3) in zip(ar_letras_staff, ar_pesos_staff, ar_letras_ambience, ar_pesos_ambience, ar_letras_food, ar_pesos_food,ar_letras_rf1, ar_pesos_rf1, ar_letras_rf2, ar_pesos_rf2, ar_letras_rf3, ar_pesos_rf3):
            #mean_simil_contex implementar el iterador que promedie para todas las letras de entrada(funcion)
            #pesos_ref = [(i+1)/(i+1) for i in range(len(pesos))]
            #print(pesos_ref)
            #val = mean_mean_sim_context(letras_staff, pesos_staff, pesos_ref, refword, list(set(refwords)-set(refword)))
            #val_staff = mean_simil_contex()
            val_rw1 = mean_mean_sim_context(letras_staff, pesos_staff , pesos_rf1, refword, list(set(refwords)-set(refword)))
            val_rw2 = mean_mean_sim_context(letras_ambience, pesos_ambience , pesos_rf2, refword, list(set(refwords)-set(refword)))
            val_rw3 = mean_mean_sim_context(letras_food, pesos_food , pesos_rf3, refword, list(set(refwords)-set(refword)))
            out.write(' '.join(letras_staff)+' --- '+str(1*val_rw1 + 1*val_rw2 + 1*val_rw3)+'\n')
            refword_vals.append(1*val_rw1 +1*val_rw2+1*val_rw3)
        result.append(refword_vals)
    return result

result = method1()
print(result[0])
print(result[1])
print(result[2])
predict_labels = []


out_labels = codecs.open('test_labels_abae.txt', 'w', 'utf-8')

for val1, val2, val3 in zip(result[0], result[1], result[2]):
    (_, index) = mayor3(val1, val2, val3)
    if index==0:
        out_labels.write('Staff\n')
        predict_labels.append('Staff')
        print('0')
    elif index==1:
        out_labels.write('Ambience\n')
        predict_labels.append('Ambience')
        print('1')
    elif index==2:
        out_labels.write('Food\n')
        predict_labels.append('Food\n')
        print('2')


domain = 'restaurant'
print('--- Results on %s domain ----' %(domain))
#test_labels = '/home/danny/gpuimp/experiments/v3_mymodel_onebyone/modified_abae_v3_mymodel_food/preprocessed_data/%s/test_label.txt' % (domain)

test_labels = '/home/danny/gpuimp/experiments/v3_mymodel_onebyone/modified_abae_v3_mymodel_staff/preprocessed_data/%s/test_label.txt' % (domain)

evaluation(open(test_labels), predict_labels, domain)

