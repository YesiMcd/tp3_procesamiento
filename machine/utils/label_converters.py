def label_to_int(string_label):
    if string_label == 'peon': return 1
    if string_label == 'torre': return 2
    if string_label == 'caballo': return 3
    if string_label == 'alfil': return 4
    if string_label == 'reina': return 5
    if string_label == 'rey': return 6
    else:
        raise Exception('Unknown class_label')


def int_to_label(class_int):
    if class_int == 1: return 'peon'
    if class_int == 2: return 'torre'
    if class_int == 3: return 'caballo'
    if class_int == 4: return 'alfil'
    if class_int == 5: return 'reina'
    if class_int == 6: return 'rey'
    else:
        raise Exception('Unknown class_label')
