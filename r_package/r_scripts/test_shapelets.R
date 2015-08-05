

require(Rdnn)

v = RProto$new("/home/alexeyche/dnn/build/left.pb")$read()$values[[1]]
vr = RProto$new("/home/alexeyche/dnn/build/right1.pb")$read()$values[[1]]
vr4 = RProto$new("/home/alexeyche/dnn/build/right4.pb")$read()$values[[1]]