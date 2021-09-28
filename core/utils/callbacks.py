# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Callback utils
"""


class Callbacks:
    """"
    YOLOv5 钩子: 处理所有注册的回调函数
    """

    # 定义可用的钩子
    _callbacks = {
        'on_pretrain_routine_start': [],
        'on_pretrain_routine_end': [],

        'on_train_start': [],
        'on_train_epoch_start': [],
        'on_train_batch_start': [],
        'optimizer_step': [],
        'on_before_zero_grad': [],
        'on_train_batch_end': [],
        'on_train_epoch_end': [],

        'on_val_start': [],
        'on_val_batch_start': [],
        'on_val_image_end': [],
        'on_val_batch_end': [],
        'on_val_end': [],

        'on_fit_epoch_end': [],  # fit = train + val
        'on_model_save': [],
        'on_train_end': [],

        'teardown': [],
    }

    def register_action(self, hook, name='', callback=None):
        """
        往回调钩子中注册一个新的动作
        参数:
            hook        动作要注册的目标钩子
            name        动作的名字, 方便之后引用
            callback    与 name 对应的回调函数
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook=None):
        """"
        返回所有注册到 hook 中的动作
        参数:
            返回 hook 中注册的所有回调, 默认返回所有 hook
        """
        if hook:
            return self._callbacks[hook]
        else:
            return self._callbacks

    def run(self, hook, *args, **kwargs):
        """
        遍历指定的 hook, 并运行其中所有动作
        参数:
            hook    要查看的 hook 名称
            args    要从 YOLOv5 中接收的参数
            kwargs  要从 YOLOv5 中接收的关键字参数
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"

        for logger in self._callbacks[hook]:
            logger['callback'](*args, **kwargs)