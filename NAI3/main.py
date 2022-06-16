from Perceptron import *


def max_selector(perceptron_list):
    max_value = perceptron_list[0].current_output
    lang = perceptron_list[0].name
    for perceptron in perceptron_list:
        if perceptron.current_output > max_value:
            max_value = perceptron.current_output
            lang = perceptron.name
    return lang


if __name__ == '__main__':
    os.chdir('testing')
    directories = os.listdir('.')
    for d in directories:
        print(d)
    perceptrons = []
    alpha = 0.5
    error = 3
    flag = True
    for d in directories:
        perceptrons.append(Perceptron(d, alpha))
    print('Start')
    while flag:

        for d in reversed(directories):

            files_list = os.listdir(os.path.abspath(d))
            os.chdir(d)
            error_sum = 0
            for f in files_list:
                for p in perceptrons:
                    p.process(f)
                    error_sum += p.error
                file_language = max_selector(perceptrons)

            os.chdir('..')
            if error_sum < error:
                flag = False
            print(f'Error {error_sum}')

    if len(files_list) == 0:
        print('No directories found')
    else:
        for f in files_list:
            print(f'File{f} ')
            for p in perceptrons:
                p.test(f)
                print(f'Perceptron {p.name} - output {p.current_output}')
            file_language = max_selector(perceptrons)
            print(f'File: {f}, Language: {file_language}')
