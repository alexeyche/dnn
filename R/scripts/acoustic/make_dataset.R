require(Rdnn)

test_licks = proto.read(spikes.path("test_licks.pb"))

test_licks = add.ts.info(test_licks, ts.info(label="0", start_time=150, duration=500))
test_licks = add.ts.info(test_licks, ts.info(label="1", start_time=740, duration=360))
test_licks = add.ts.info(test_licks, ts.info(label="2", start_time=1250, duration=550))
test_licks = add.ts.info(test_licks, ts.info(label="3", start_time=1850, duration=650))
test_licks = add.ts.info(test_licks, ts.info(label="4", start_time=2550, duration=250))
test_licks = add.ts.info(test_licks, ts.info(label="5", start_time=2800, duration=350))
test_licks = add.ts.info(test_licks, ts.info(label="6", start_time=3200, duration=900))
test_licks = add.ts.info(test_licks, ts.info(label="7", start_time=4150, duration=520))
test_licks = add.ts.info(test_licks, ts.info(label="8", start_time=4670, duration=630))
test_licks = add.ts.info(test_licks, ts.info(label="9", start_time=5300, duration=900))

test_licks = add.to.spikes(test_licks, test_licks)
proto.write(test_licks, spikes.path("work_licks_eval.pb"))

while (TRUE) {
    max_t = max(sapply(test_licks$values, function(x) if(length(x)>0) { max(x)} else {0}))
    if (max_t > 60000) {
        break
    }
    test_licks = add.to.spikes(test_licks, test_licks)
}

proto.write(test_licks, spikes.path("work_licks.pb"))

