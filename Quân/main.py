import turtle as t

# t.fd(100)
# t.shape("turtle")

t.pendown()
def square(d):
    for i in range(4):
        t.fd(d)
        t.left(90)

square(100)

t.listen()
t.exitonclick()