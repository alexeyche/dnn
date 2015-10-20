
require(Rdnn)

dt = 5
sim_length = 60000 # длина тестового семпла, мы сгенерим сначала короткий семпл из двух классов, а потом размножим его до указанного времени
neurons = 100
sample_gap = 100 # расстояние между двумя семплами классов, семплы конкатенируются в один длинный временной ряд, для удобства симуляции 

ts = time.series(
    matrix(c(1:100, 100:1), nrow=1)  # довольно простой временной ряд, нарисуй plot(ts), станет ясно
  , ts.info(c("1","2"), c(100, 200))   # здесь описываются лейблы классов и времена когда они заканчиваются
)


res_sp = empty.spikes(neurons) # готовит лист спайков для N нейронов

while(TRUE) {
    sp = intercept.data.to.spikes(
        ts
        , neurons
        , 1
        , dt
        , sample_gap
    )
    res_sp = cat.spikes(res_sp, sp)
    if(tail(res_sp$ts_info$labels_timeline, n=1)>sim_length) {
        break
    }
}

proto.write(res_sp, spikes.path("test_spikes.pb"))

plot(res_sp, i=1) # нарисовать спайки для семпла 1
