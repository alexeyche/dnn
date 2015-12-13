
require(rjson)

user.json = fromJSON(readConst(user.json.file()))
user.env = user.json[[ Sys.getenv("USER") ]]
if(!is.null(user.env)) {
    do.call(Sys.setenv, user.env)
}