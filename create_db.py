import rethinkdb as r

r.connect("localhost", 28015).repl()

r.db_create("robot_does_x").run()
r.db("robot_does_x").create("train_output").run()
