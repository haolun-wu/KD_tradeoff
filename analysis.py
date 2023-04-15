import numpy as np
import matplotlib.pyplot as plt

gt = np.load('value_save/gt_label.npy')
t_value = np.load('value_save/teacher_label_128_100.npy')
s_value = np.load('value_save/student_label_16_100.npy')

# there are 100k data. I choose part of them as an example for visualization
gt_sample = gt[:10000]
t_value_sample = t_value[:10000]
s_value_sample = s_value[:10000]

gt_sample, t_value_sample, s_value_sample = zip(*sorted(zip(gt_sample, t_value_sample, s_value_sample)))

plt.rcParams["figure.figsize"] = (4.5 * 4, 4.5)
# fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4),
#                         constrained_layout=True) #, sharex=True, sharey=True)


plt.plot(gt_sample, label='GT')
plt.plot(t_value_sample, label='Teacher')
plt.plot(s_value_sample, label='Student (Teacher)')
# plt.plot(np.abs(np.array(s_value_sample)-np.array(t_value_sample)), label='S-T Agreement')
plt.legend()
plt.show()


agree_student_teacher = np.abs(np.array(s_value_sample)-np.array(t_value_sample))
confidence_teacher = np.abs(np.array(gt_sample)-np.array(t_value_sample))
plt.plot(agree_student_teacher, label="Diff of S-T")
plt.plot(confidence_teacher, label='Diff of T-GT')
plt.legend()
plt.show()