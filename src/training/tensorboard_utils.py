from tensorflow import summary, convert_to_tensor


def write_tensorboard_logs(writer, model, step, reward):
    if step % 100 == 0:
        summary.scalar(name="reward", data=reward, step=step)
        dqn_variable = model.trainable_variables
        summary.histogram(
            name="dqn_variables",
            data=convert_to_tensor(dqn_variable[0]),
            step=step,
        )
        writer.flush()
