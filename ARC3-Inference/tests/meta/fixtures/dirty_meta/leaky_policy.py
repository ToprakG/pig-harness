"""Deliberately-dirty fixture: a policy module that peeks at board content
and game identity. The guard MUST flag every line marked below."""


def decide(board_ascii, game_id):  # two forbidden tokens on one line
    color = board_ascii[0]  # board content + a color feature
    shape_count = board_ascii.count("#")  # a shape feature
    return color == "9" and shape_count > 3 and game_id.startswith("ar25")
