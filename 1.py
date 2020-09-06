# actions = ['fold', 'check', 'call' , 'allin']
actions = []
for raise_amount in range(25,301,25):
    actions.append('raise_{}_pot'.format(raise_amount))
# for raise_amount in range(33,67,33):
#     actions.append('raise_{}_pot'.format(raise_amount))
print(actions)
for i in range(len(actions)):
    print(actions[i][0:5],(int)(actions[i][6:-4])/100)

