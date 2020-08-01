def test_kwargs(first, *args, **kwargs):
    print('Required argument: ', first)
    print(args)
    print(kwargs)


test_kwargs(1, 2, 3, 4, k1=5, k2=6)
